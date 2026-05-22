#!/usr/bin/env python3
"""Extract per-run dollar spending from mini-swe-agent trajectories.

Reads `trajectories/<model_stem>_<algo>.json`, pulls
`info.model_stats.instance_cost` from each, and writes
`results/cost/cost_usd.csv` with 10 algorithm rows x 5 model-stem columns.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "trajectories"
COST_DIR = ROOT / "results" / "cost"

ALGORITHMS = [
    "autoencoder", "bert_tiny", "efficientvit", "minideit", "minilm",
    "mobilenetv4", "tinyclip_text", "tinyclip_vision_single",
    "nightvision", "wami",
]
MODEL_STEMS = ["minimax", "kimi", "opus", "gpt", "gemini"]


def extract_cost(path: Path) -> float | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    return d.get("info", {}).get("model_stats", {}).get("instance_cost")


def main() -> None:
    table: dict[tuple[str, str], float] = {}
    zero_cells = []

    for algo in ALGORITHMS:
        for stem in MODEL_STEMS:
            path = TRAJ / f"{stem}_{algo}.json"
            if not path.exists():
                continue
            cost = extract_cost(path)
            if cost is None:
                continue
            table[(algo, stem)] = cost
            if cost == 0.0:
                zero_cells.append((algo, stem))

    # per-model (column) and per-algorithm (row) breakdowns
    per_model = {s: 0.0 for s in MODEL_STEMS}
    per_algo = {a: 0.0 for a in ALGORITHMS}
    for (algo, stem), v in table.items():
        per_model[stem] += v
        per_algo[algo] += v
    grand = sum(per_model.values())

    COST_DIR.mkdir(parents=True, exist_ok=True)
    out = COST_DIR / "cost_usd.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", *MODEL_STEMS, "total"])
        for algo in ALGORITHMS:
            row = [algo]
            for stem in MODEL_STEMS:
                v = table.get((algo, stem))
                row.append(f"{v:.4f}" if v is not None else "FAIL")
            row.append(f"{per_algo[algo]:.4f}")
            w.writerow(row)
        w.writerow(["total", *(f"{per_model[s]:.4f}" for s in MODEL_STEMS), f"{grand:.4f}"])

    total = len(ALGORITHMS) * len(MODEL_STEMS)
    parsed = len(table)

    print(f"wrote {out}")
    print(f"{parsed}/{total} trajectories parsed")

    print("\nSpend per model:")
    for stem in MODEL_STEMS:
        print(f"  {stem:10s}  ${per_model[stem]:8.2f}")
    print(f"  {'TOTAL':10s}  ${grand:8.2f}")

    print("\nSpend per algorithm:")
    for algo in ALGORITHMS:
        print(f"  {algo:26s}  ${per_algo[algo]:8.2f}")
    if zero_cells:
        print(f"\n{len(zero_cells)} cells report $0.00 "
              "(likely deepinfra — LiteLLM doesn't price those):")
        for a, s in zero_cells[:10]:
            print(f"  $0.00  {a}/{s}")


if __name__ == "__main__":
    main()
