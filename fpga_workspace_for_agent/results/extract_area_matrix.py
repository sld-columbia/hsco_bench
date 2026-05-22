#!/usr/bin/env python3
"""Build area-utilization matrices (one CSV per metric) across the campaign.

Rows: algorithms.  Cols: model stems.  Cells: float % or "FAIL" when no bitstream.

Inputs that feed avg_util are reported alongside so the headline is auditable.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
AREA_DIR = RESULTS / "area"

ALGORITHMS = [
    "autoencoder", "bert_tiny", "efficientvit", "minideit", "minilm",
    "mobilenetv4", "tinyclip_text", "tinyclip_vision_single",
    "nightvision", "wami",
]
MODEL_STEMS = ["minimax", "kimi", "opus", "gpt", "gemini"]

# xc7vx485t (VC707) 
AVAIL_LUTS   = 303600
AVAIL_FFS    = 607200
AVAIL_DSPS   = 2800
AVAIL_RAMB18 = 2060
AVAIL_RAMB36 = 1030
AVAIL_BRAM_KB = 37080  # 1030 RAMB36 * 36 Kb


def rpt_path(algo: str, stem: str) -> Path:
    return (ROOT / f"{algo}_vc707_{stem}"
            / "vivado/esp-xilinx-vc707-xc7vx485t.runs/impl_1/top_utilization_placed.rpt")


def _first_int(pattern: str, text: str, field_index: int) -> int:
    """grep+sed+cut equivalent: first line matching pattern → squeeze spaces → split → take field_index (1-based)."""
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise ValueError(f"pattern not found: {pattern!r}")
    parts = re.sub(r"\s+", " ", m.group(0)).strip().split(" ")
    return int(parts[field_index - 1])


def parse(rpt: Path) -> dict[str, float]:
    text = rpt.read_text()
    # field indices mirror the reference script
    luts     = _first_int(r"^.*Slice LUTs.*$",             text, 5)
    ffs      = _first_int(r"^.*Register as Flip Flop.*$",  text, 7)
    dsps     = _first_int(r"^.*DSPs.*$",                   text, 4)
    ramb18   = _first_int(r"^.*RAMB18 .*$",                text, 4)
    ramb36   = _first_int(r"^.*RAMB36/FIFO\*.*$",          text, 4)

    bram_kb = ramb36 * 36 + ramb18 * 18
    pct_luts = luts * 100 / AVAIL_LUTS
    pct_ffs  = ffs  * 100 / AVAIL_FFS
    pct_dsps = dsps * 100 / AVAIL_DSPS
    pct_bram = bram_kb * 100 / AVAIL_BRAM_KB
    avg = (pct_luts + pct_ffs + pct_dsps + pct_bram) / 4
    return {
        "avg_util_pct": avg,
        "luts_pct":     pct_luts,
        "ffs_pct":      pct_ffs,
        "dsps_pct":     pct_dsps,
        "bram_pct":     pct_bram,
    }


def write_matrix(filename: str, metric: str, table: dict[tuple[str, str], dict]) -> None:
    AREA_DIR.mkdir(parents=True, exist_ok=True)
    path = AREA_DIR / filename
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", *MODEL_STEMS])
        for algo in ALGORITHMS:
            row = [algo]
            for stem in MODEL_STEMS:
                v = table.get((algo, stem), {}).get(metric)
                row.append(f"{v:.2f}" if isinstance(v, float) else "FAIL")
            w.writerow(row)
    print(f"wrote {path}")


def main() -> None:
    table: dict[tuple[str, str], dict] = {}
    missing = []
    for algo in ALGORITHMS:
        for stem in MODEL_STEMS:
            rpt = rpt_path(algo, stem)
            if not rpt.exists():
                missing.append((algo, stem))
                continue
            try:
                table[(algo, stem)] = parse(rpt)
            except Exception as e:
                print(f"[warn] parse failed for {algo}/{stem}: {e}")
                missing.append((algo, stem))

    # headline first, then the four inputs that feed it
    write_matrix("area_avg_util.csv", "avg_util_pct", table)
    write_matrix("area_luts_pct.csv", "luts_pct",     table)
    write_matrix("area_ffs_pct.csv",  "ffs_pct",      table)
    write_matrix("area_dsps_pct.csv", "dsps_pct",     table)
    write_matrix("area_bram_pct.csv", "bram_pct",     table)

    print(f"\n{len(table)} workspaces parsed, {len(missing)} marked FAIL")
    if missing:
        for a, s in missing[:10]:
            print(f"  FAIL  {a}/{s}")
        if len(missing) > 10:
            print(f"  ... ({len(missing) - 10} more)")


if __name__ == "__main__":
    main()
