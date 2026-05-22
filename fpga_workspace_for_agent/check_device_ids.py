#!/usr/bin/env python3
"""
check_device_ids.py — validate device_id assignments in an FPGA workspace.

Usage:
    check_device_ids.py <workspace_path> [--fix-ids]

For the given workspace this script:
  1. Parses esp_xilinx-vc707-xc7vx485t_defconfig to find which accelerators
     are scheduled for install (each TILE line containing 'acc <NAME>').
  2. For each accelerator, reads device_id from BOTH:
       <prefix>_workspace/<name>/<name>.xml          (attribute device_id="...")
       <prefix>_workspace/<prefix>_sw_driver/<prefix>_cfg.h  (#define <PREFIX>_DEV_ID 0x...)
     and verifies they match (catches typos).
  3. Validates that every workspace accelerator's id is:
       (a) unique within the workspace,
       (b) not in the ESP-reserved range (0x000-0x0FF, 0x100, 0x144, 0x222),
       (c) not already claimed by another project's accelerator already
           installed under $ESP_ROOT/tech/virtex7/acc/.
  4. Reports any issues with a structured summary.
  5. With --fix-ids, automatically reassigns colliding/reserved IDs to the
     first free value in 0x201-0x27F (this project's reserved range), updating
     BOTH the XML and the cfg.h. Lex-first accelerator name keeps its id;
     subsequent collisions are bumped.

Exit codes:
    0  all device IDs are clean
    1  collisions / reserved hits remain (after fix attempt, if --fix-ids set)
    2  usage or file-not-found error
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Reserved IDs sourced from README "Device ID Allocation" section.
RESERVED_RANGE: Set[int] = set(range(0x000, 0x100))   # ESP infra + stock
RESERVED_POINT: Dict[int, str] = {
    0x100: "NVDLA",
    0x144: "sinkhorn",
    0x222: "svd",
}
RESERVED_IDS: Set[int] = RESERVED_RANGE | set(RESERVED_POINT)

# This project's reserved allocation range (inclusive at both ends).
PROJECT_RANGE = range(0x200, 0x280)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass
class AccelInfo:
    name: str           # lowercase, e.g. "dense_sysc_catapult"
    xml_path: Path
    cfg_path: Path
    xml_id: int
    cfg_id: int

    @property
    def id(self) -> int:
        # Canonical id (validated against cfg_id elsewhere).
        return self.xml_id


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def die(msg: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def reserved_label(rid: int) -> str:
    if rid in RESERVED_POINT:
        return RESERVED_POINT[rid]
    if rid in RESERVED_RANGE:
        return "ESP infrastructure / stock accelerator"
    return "unknown"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_TILE_RE = re.compile(r"^TILE_\d+_\d+\s*=\s*\d+\s+acc\s+(\S+)")
_XML_ID_RE = re.compile(r'device_id\s*=\s*"([0-9a-fA-F]+)"')
_XML_NAME_RE = re.compile(r'<accelerator\s+name\s*=\s*"([^"]+)"')
_CFG_ID_RE = re.compile(r"#define\s+\w+_DEV_ID\s+0x([0-9a-fA-F]+)")


def parse_defconfig(defconfig: Path) -> List[str]:
    """Return ordered list of unique accelerator names (lowercase)."""
    if not defconfig.is_file():
        die(f"defconfig not found: {defconfig}")
    seen: Set[str] = set()
    out: List[str] = []
    for line in defconfig.read_text().splitlines():
        m = _TILE_RE.match(line)
        if not m:
            continue
        n = m.group(1).lower()
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def locate_files(ws: Path, name: str) -> Tuple[Path, Path]:
    """Return (xml_path, cfg_h_path) for the accelerator named `name`."""
    prefix = re.sub(r"_sysc_catapult$", "", name)
    xml = ws / f"{prefix}_workspace" / name / f"{name}.xml"
    cfg = ws / f"{prefix}_workspace" / f"{prefix}_sw_driver" / f"{prefix}_cfg.h"
    return xml, cfg


def parse_xml_id(xml: Path) -> int:
    if not xml.is_file():
        die(f"XML not found: {xml}")
    m = _XML_ID_RE.search(xml.read_text())
    if not m:
        die(f"no device_id attribute found in {xml}")
    return int(m.group(1), 16)


def parse_cfg_id(cfg: Path) -> int:
    if not cfg.is_file():
        die(f"cfg.h not found: {cfg}")
    m = _CFG_ID_RE.search(cfg.read_text())
    if not m:
        die(f"no '#define ..._DEV_ID 0x..' line found in {cfg}")
    return int(m.group(1), 16)


def scan_cross_project(esp_root: Path, our_accels: Set[str]) -> Dict[int, str]:
    """Scan $ESP_ROOT/tech/virtex7/acc/*/*.xml for device IDs claimed by
    accelerators OTHER than the ones we are about to (re)install."""
    result: Dict[int, str] = {}
    acc_root = esp_root / "tech" / "virtex7" / "acc"
    if not acc_root.is_dir():
        warn(f"cross-project scan skipped: {acc_root} does not exist")
        return result
    for xml in sorted(acc_root.glob("*/*.xml")):
        try:
            text = xml.read_text()
        except Exception as e:
            warn(f"could not read {xml}: {e}")
            continue
        nm = _XML_NAME_RE.search(text)
        idm = _XML_ID_RE.search(text)
        if not (nm and idm):
            continue
        ext_name = nm.group(1).lower()
        if ext_name in our_accels:
            # We will overwrite this on install — not a collision source.
            continue
        ext_id = int(idm.group(1), 16)
        # If multiple external accels share an id (other projects' bug),
        # remember the last one we saw — it's enough to mark the id taken.
        result[ext_id] = ext_name
    return result


# ---------------------------------------------------------------------------
# auto-fix
# ---------------------------------------------------------------------------

def auto_fix(accels: List[AccelInfo],
             cross: Dict[int, str]) -> List[Tuple[str, int, int]]:
    """Mutate accels[].xml_id and .cfg_id in place to resolve collisions.

    Lex-smallest accelerator name keeps its current id (if not reserved /
    cross-claimed). Subsequent accels claiming an already-used id, or
    landing on a reserved/cross-claimed id, are reassigned to the first
    free value inside PROJECT_RANGE.

    Returns list of (name, old_id, new_id) for accelerators actually moved.
    """
    final_ids: Set[int] = set(cross.keys()) | RESERVED_IDS
    fixes: List[Tuple[str, int, int]] = []
    for acc in sorted(accels, key=lambda a: a.name):
        if acc.id in final_ids:
            new_id = next((c for c in PROJECT_RANGE if c not in final_ids), None)
            if new_id is None:
                die(f"no free id in 0x{PROJECT_RANGE.start:x}-"
                    f"0x{PROJECT_RANGE.stop - 1:x} for {acc.name}")
            fixes.append((acc.name, acc.id, new_id))
            acc.xml_id = new_id
            acc.cfg_id = new_id
        final_ids.add(acc.id)
    return fixes


def apply_fix_xml(path: Path, new_id: int) -> None:
    text = path.read_text()
    new_text, n = _XML_ID_RE.subn(rf'device_id="{new_id:x}"', text)
    if n == 0:
        die(f"could not patch device_id attribute in {path}")
    path.write_text(new_text)


def apply_fix_cfg(path: Path, new_id: int) -> None:
    text = path.read_text()
    # Preserve the original whitespace before the value by capturing the prefix.
    new_text, n = re.subn(
        r"(#define\s+\w+_DEV_ID\s+0x)[0-9a-fA-F]+",
        rf"\g<1>{new_id:x}",
        text,
    )
    if n == 0:
        die(f"could not patch *_DEV_ID #define in {path}")
    path.write_text(new_text)


# ---------------------------------------------------------------------------
# detection / reporting
# ---------------------------------------------------------------------------

def detect_issues(accels: List[AccelInfo],
                  cross: Dict[int, str]) -> List[str]:
    issues: List[str] = []
    seen_ids: Dict[int, str] = {}
    for acc in accels:
        if acc.xml_id != acc.cfg_id:
            issues.append(
                f"  {acc.name}: XML id 0x{acc.xml_id:x} does not match "
                f"cfg.h id 0x{acc.cfg_id:x} — files out of sync"
            )
        if acc.id in RESERVED_IDS:
            issues.append(
                f"  {acc.name}: 0x{acc.id:x} is RESERVED ({reserved_label(acc.id)})"
            )
        if acc.id in seen_ids:
            issues.append(
                f"  {acc.name}: 0x{acc.id:x} collides with workspace accel "
                f"'{seen_ids[acc.id]}'"
            )
        else:
            seen_ids[acc.id] = acc.name
        if acc.id in cross:
            issues.append(
                f"  {acc.name}: 0x{acc.id:x} collides with externally-installed "
                f"accelerator '{cross[acc.id]}' "
                f"(under $ESP_ROOT/tech/virtex7/acc/)"
            )
    return issues


def print_summary(accels: List[AccelInfo], cross: Dict[int, str]) -> None:
    print("Workspace accelerators:")
    width = max((len(a.name) for a in accels), default=20)
    for acc in sorted(accels, key=lambda a: a.name):
        marker = "ok" if acc.xml_id == acc.cfg_id else "MISMATCH"
        print(f"  {acc.name:<{width}}  xml=0x{acc.xml_id:x}  "
              f"cfg=0x{acc.cfg_id:x}  [{marker}]")
    if cross:
        print("\nExternally-claimed IDs (other projects in $ESP_ROOT/tech):")
        for cid, cname in sorted(cross.items()):
            print(f"  0x{cid:x}  {cname}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate FPGA accelerator device_id assignments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("workspace_path", type=Path,
                    help="path to the workspace directory")
    ap.add_argument("--fix-ids", action="store_true",
                    help="auto-reassign colliding / reserved IDs into "
                         "0x201-0x27F (modifies XML and cfg.h)")
    args = ap.parse_args()

    ws: Path = args.workspace_path.resolve()
    if not ws.is_dir():
        die(f"workspace not found: {ws}")

    defconfig = ws / "esp_xilinx-vc707-xc7vx485t_defconfig"
    names = parse_defconfig(defconfig)
    print(f"=== Device ID validation: {ws.name} ===")

    if not names:
        print("No accelerators declared in defconfig — nothing to validate.")
        sys.exit(0)

    print(f"Accelerators discovered: {len(names)}")
    print()

    accels: List[AccelInfo] = []
    for name in names:
        xml_p, cfg_p = locate_files(ws, name)
        accels.append(AccelInfo(
            name=name,
            xml_path=xml_p,
            cfg_path=cfg_p,
            xml_id=parse_xml_id(xml_p),
            cfg_id=parse_cfg_id(cfg_p),
        ))

    esp_root = Path(os.environ["ESP_ROOT"])
    cross = scan_cross_project(esp_root, {a.name for a in accels})

    print_summary(accels, cross)
    print()

    issues = detect_issues(accels, cross)
    if not issues:
        print("=== PASS: all device IDs are clean ===")
        sys.exit(0)

    print(f"=== {len(issues)} issue(s) found ===")
    for line in issues:
        print(line)
    print()

    if not args.fix_ids:
        print("Re-run with --fix-ids to automatically reassign colliding IDs "
              "into 0x201-0x27F.")
        sys.exit(1)

    # --------- auto-fix path ---------
    fixes = auto_fix(accels, cross)
    if not fixes:
        # Happens when the only issue is xml/cfg mismatch (cannot be auto-fixed
        # because we cannot guess which of the two is the intended truth).
        print("=== --fix-ids cannot resolve the issues above (likely "
              "XML/cfg.h mismatch — please reconcile by hand) ===")
        sys.exit(1)

    print("Applying fixes:")
    for name, old, new in fixes:
        acc = next(a for a in accels if a.name == name)
        apply_fix_xml(acc.xml_path, new)
        apply_fix_cfg(acc.cfg_path, new)
        print(f"  {name}:  0x{old:x} -> 0x{new:x}")
        print(f"    XML : {acc.xml_path}")
        print(f"    cfg : {acc.cfg_path}")
    print()

    # Re-validate from disk to make sure the patches stuck.
    for acc in accels:
        acc.xml_id = parse_xml_id(acc.xml_path)
        acc.cfg_id = parse_cfg_id(acc.cfg_path)
    remaining = detect_issues(accels, cross)
    if remaining:
        print(f"=== FAIL: {len(remaining)} issue(s) remain after auto-fix ===")
        for line in remaining:
            print(line)
        sys.exit(1)

    print("=== PASS: all device IDs are clean (after auto-fix) ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
