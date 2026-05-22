#!/usr/bin/env python3
"""Recompute area metrics using the 3-resource formula.

Reads the per-resource CSV matrices already produced by extract_area_matrix.py
and outputs CSVs:
  - area_logic_cells_pct.csv  (= luts_pct, since 303600*1.6 == 485760)
  - area_bram_pct.csv         (unchanged)
  - area_dsps_pct.csv         (unchanged)
  - area_avg_util.csv         (logic_cells + bram + dsp) / 3

All outputs go into this same area_compile/ directory.
"""

from __future__ import annotations

import csv
from pathlib import Path

AREA_DIR = Path(__file__).resolve().parent.parent  # .../results/area
OUT_DIR = Path(__file__).resolve().parent           # .../results/area/area_compile

# Source CSVs produced by extract_area_matrix.py
LUTS_CSV = AREA_DIR / "area_luts_pct.csv"
BRAM_CSV = AREA_DIR / "area_bram_pct.csv"
DSPS_CSV = AREA_DIR / "area_dsps_pct.csv"


def read_matrix(path: Path) -> tuple[list[str], list[str], dict[tuple[str, str], float]]:
    """Return (algorithms, model_stems, {(algo,stem): value})."""
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        stems = header[1:]
        algos = []
        data: dict[tuple[str, str], float] = {}
        for row in reader:
            if not row:
                continue
            algo = row[0]
            algos.append(algo)
            for stem, cell in zip(stems, row[1:]):
                if cell != "FAIL":
                    data[(algo, stem)] = float(cell)
    return algos, stems, data


def write_matrix(path: Path, algos: list[str], stems: list[str],
                 data: dict[tuple[str, str], float]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", *stems])
        for algo in algos:
            row = [algo]
            for stem in stems:
                v = data.get((algo, stem))
                row.append(f"{v:.2f}" if v is not None else "FAIL")
            w.writerow(row)
    print(f"wrote {path}")


def main() -> None:
    algos, stems, luts = read_matrix(LUTS_CSV)
    _,     _,     bram = read_matrix(BRAM_CSV)
    _,     _,     dsps = read_matrix(DSPS_CSV)

    # logic_cells_pct = luts_pct  (since AVAIL_LOGIC_CELLS = AVAIL_LUTS * 1.6)
    logic_cells = dict(luts)

    # New avg: (logic_cells% + bram% + dsp%) / 3
    avg: dict[tuple[str, str], float] = {}
    for key in logic_cells:
        if key in bram and key in dsps:
            avg[key] = (logic_cells[key] + bram[key] + dsps[key]) / 3

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_matrix(OUT_DIR / "area_logic_cells_pct.csv", algos, stems, logic_cells)
    write_matrix(OUT_DIR / "area_bram_pct.csv",        algos, stems, bram)
    write_matrix(OUT_DIR / "area_dsps_pct.csv",        algos, stems, dsps)
    write_matrix(OUT_DIR / "area_avg_util.csv",        algos, stems, avg)

    total = len(avg)
    fail = sum(1 for a in algos for s in stems if (a, s) not in avg)
    print(f"\n{total} entries computed, {fail} marked FAIL")


if __name__ == "__main__":
    main()
