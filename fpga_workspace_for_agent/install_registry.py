#!/usr/bin/env python3
"""
install_registry.py — track and clean per-workspace accelerator installs
into the shared ESP tech tree ($ESP_ROOT/tech/virtex7/acc/).

Without a registry, the ESP tech tree fills up with leftover accelerators
and there is no record of which workspace put what there. This tool keeps a
per-workspace JSON manifest at:

    <workspace>/.installed_accelerators.json

so you can later list or remove just the installs that THIS workspace owns,
without disturbing other projects sharing the same tech tree.

Subcommands:

    record       <workspace> <accel_name>          record (or refresh) one install
    list         <workspace>                       pretty-print the manifest
    cleanup      <workspace> [--dry-run]           remove this workspace's installs
                                                   and archive the manifest
    cleanup-all  [--dry-run]                       cleanup every workspace under
                                                   fpga_workspace_for_agent/

After a successful (non-dry-run) cleanup, the manifest is renamed to
    .installed_accelerators.cleaned-YYYYMMDD-HHMMSS.json
so an audit trail of what was removed (and when) is preserved.
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

# This script lives at the root of fpga_workspace_for_agent/, so its parent
# is the directory we scan for cleanup-all.
ROOT_DIR = Path(__file__).resolve().parent

MANIFEST_NAME = ".installed_accelerators.json"
ARCHIVE_PREFIX = ".installed_accelerators.cleaned-"

# Cleanup safety floor: never remove anything from the ESP tech tree whose
# recorded device_id is below this value, even if it appears in our manifest.
# Protects ESP infrastructure (0x000-0x0FF), stock accelerators (NVDLA at
# 0x100, sinkhorn at 0x144, etc.), and accelerators that may belong to a
# DIFFERENT project that happens to share the same machine.
CLEANUP_FLOOR = 0x200


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def die(msg: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def now_filename_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def manifest_path(ws: Path) -> Path:
    return ws / MANIFEST_NAME


def load_manifest(ws: Path) -> Dict[str, Any]:
    p = manifest_path(ws)
    if not p.is_file():
        return {"workspace": str(ws), "installed": []}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        die(f"manifest at {p} is corrupt: {e}")


def save_manifest(ws: Path, data: Dict[str, Any]) -> None:
    manifest_path(ws).write_text(json.dumps(data, indent=2) + "\n")


def detect_device_id(ws: Path, accel: str) -> str:
    """Best-effort: scrape the XML to remember which device_id this install used."""
    if accel.endswith("_sysc_catapult"):
        prefix = accel[: -len("_sysc_catapult")]
    else:
        prefix = accel
    xml = ws / f"{prefix}_workspace" / accel / f"{accel}.xml"
    if not xml.is_file():
        return "?"
    m = re.search(r'device_id\s*=\s*"([0-9a-fA-F]+)"', xml.read_text())
    return f"0x{int(m.group(1), 16):x}" if m else "?"


def esp_tech_path(accel: str) -> Path:
    esp_root = Path(os.environ["ESP_ROOT"])
    return esp_root / "tech" / "virtex7" / "acc" / accel


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_record(ws: Path, accel: str) -> int:
    """Record (or refresh, by name) a single install. Idempotent per accel name."""
    data = load_manifest(ws)
    # Dedupe: drop any prior entry for the same accel name.
    data["installed"] = [e for e in data["installed"] if e.get("name") != accel]
    data["installed"].append({
        "name":         accel,
        "tech_path":    str(esp_tech_path(accel)),
        "device_id":    detect_device_id(ws, accel),
        "installed_at": now_iso(),
    })
    # Sort for deterministic on-disk diff.
    data["installed"].sort(key=lambda e: e["name"])
    save_manifest(ws, data)
    print(f"registered install: {accel}  ({manifest_path(ws).name})")
    return 0


def cmd_list(ws: Path) -> int:
    data = load_manifest(ws)
    print(f"Workspace: {data.get('workspace', ws)}")
    print(f"Manifest:  {manifest_path(ws)}")
    installs: List[Dict[str, Any]] = data.get("installed", [])
    if not installs:
        print("  (no installs recorded)")
        return 0
    width = max((len(e["name"]) for e in installs), default=20)
    print(f"  {'name':<{width}}  device     installed_at         tech_path")
    for e in installs:
        print(f"  {e['name']:<{width}}  "
              f"{e.get('device_id', '?'):<8}  "
              f"{e.get('installed_at', '?'):<19}  "
              f"{e.get('tech_path', '?')}")
    return 0


def cmd_cleanup(ws: Path, dry_run: bool = False, archive: bool = True) -> int:
    p = manifest_path(ws)
    if not p.is_file():
        print(f"no manifest at {p} — nothing to clean")
        return 0

    data = load_manifest(ws)
    installs: List[Dict[str, Any]] = data.get("installed", [])
    if not installs:
        print(f"manifest is empty — nothing to clean")
        return 0

    print(f"workspace: {ws}")
    print(f"manifest:  {p}")
    print(f"entries:   {len(installs)}")
    print()

    removed = 0
    skipped = 0
    refused = 0
    for e in installs:
        tp = Path(e["tech_path"])
        raw_id = e.get("device_id", "?")
        label = f"{e['name']} ({raw_id})  ->  {tp}"

        # Safety gate: refuse to delete anything outside our project's range,
        # even if recorded in our manifest. Protects ESP infrastructure and
        # other projects that share this machine.
        try:
            did = int(raw_id, 16) if raw_id != "?" else None
        except ValueError:
            did = None
        if did is None or did < CLEANUP_FLOOR:
            print(f"  [REFUSE]   {label}  (device_id < 0x{CLEANUP_FLOOR:x} or unknown — refusing to remove)")
            refused += 1
            continue

        if not tp.exists():
            print(f"  [skip]     {label}  (already gone)")
            skipped += 1
            continue
        if dry_run:
            print(f"  [dry-run]  {label}")
        else:
            shutil.rmtree(tp)
            print(f"  [removed]  {label}")
        removed += 1

    print()
    if dry_run:
        print(f"DRY RUN: would remove {removed} dir(s); "
              f"{skipped} already gone; {refused} refused (id < 0x{CLEANUP_FLOOR:x})")
        print(f"(re-run without --dry-run to actually remove)")
        return 0

    if archive:
        # Archive the manifest so we keep an audit trail.
        archive_path = ws / f"{ARCHIVE_PREFIX}{now_filename_stamp()}.json"
        p.rename(archive_path)
        archive_msg = f"manifest archived: {archive_path.name}"
    else:
        # Caller (e.g. check_workspace.sh's pre-install cleanup) doesn't
        # want audit files piling up — just delete the manifest.
        p.unlink()
        archive_msg = f"manifest removed (no archive)"
    print(f"removed {removed} dir(s); {skipped} already gone; "
          f"{refused} refused (id < 0x{CLEANUP_FLOOR:x})")
    print(archive_msg)
    return 0


def cmd_cleanup_all(dry_run: bool = False, archive: bool = True) -> int:
    workspaces = sorted(
        c for c in ROOT_DIR.iterdir()
        if c.is_dir() and (c / MANIFEST_NAME).is_file()
    )
    if not workspaces:
        print(f"no workspaces with active manifests found under {ROOT_DIR}")
        return 0

    print(f"found {len(workspaces)} workspace(s) with manifests:")
    for w in workspaces:
        print(f"  - {w.name}")
    print()

    rc = 0
    for w in workspaces:
        print("=" * 70)
        print(f"== {w.name}")
        print("=" * 70)
        rc |= cmd_cleanup(w, dry_run=dry_run, archive=archive)
        print()
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p = p.resolve()
    if not p.is_dir():
        die(f"workspace not found: {p}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("record",      help="record (or refresh) an install")
    p.add_argument("workspace",  type=Path)
    p.add_argument("accel_name", type=str)

    p = sub.add_parser("list",        help="list installs in a workspace")
    p.add_argument("workspace",  type=Path)

    p = sub.add_parser("cleanup",     help="remove this workspace's installs from the ESP tech tree")
    p.add_argument("workspace",  type=Path)
    p.add_argument("--dry-run",   action="store_true",
                   help="show what would be removed; do not modify anything")
    p.add_argument("--no-archive", action="store_true",
                   help="delete manifest instead of renaming it to .cleaned-*.json "
                        "(useful when cleanup is part of a larger flow)")

    p = sub.add_parser("cleanup-all", help="cleanup every workspace under fpga_workspace_for_agent/")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--no-archive", action="store_true",
                   help="delete manifests instead of archiving (per-workspace)")

    args = ap.parse_args()

    if args.cmd == "record":
        return cmd_record(_ensure_dir(args.workspace), args.accel_name)
    if args.cmd == "list":
        return cmd_list(_ensure_dir(args.workspace))
    if args.cmd == "cleanup":
        return cmd_cleanup(_ensure_dir(args.workspace),
                           dry_run=args.dry_run,
                           archive=not args.no_archive)
    if args.cmd == "cleanup-all":
        return cmd_cleanup_all(dry_run=args.dry_run,
                               archive=not args.no_archive)

    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
