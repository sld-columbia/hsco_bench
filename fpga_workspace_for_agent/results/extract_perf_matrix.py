#!/usr/bin/env python3
"""Build a performance matrix (cycles) from on-FPGA run logs.

Reads `results/fpga_runs/<algo>_<model_stem>.log`, pulls the cycle count from
the `esp_monitor 32-bit` reporting line, and writes `results/perf/perf_cycles.csv`
with 10 algorithm rows x 5 model-stem columns. Missing or incomplete runs → FAIL.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPGA_RUNS = ROOT / "results" / "fpga_runs"
PERF_DIR = ROOT / "results" / "perf"

ALGORITHMS = [
    "autoencoder", "bert_tiny", "efficientvit", "minideit", "minilm",
    "mobilenetv4", "tinyclip_text", "tinyclip_vision_single",
    "nightvision", "wami",
]
MODEL_STEMS = ["minimax", "kimi", "opus", "gpt", "gemini"]

# Match any reporting line tagged with `esp_monitor 32-bit` — the cycle count
# can appear either before (e.g. "Text encoding time: N cycles (esp_monitor 32-bit)")
# or after the tag (e.g. "Total execution time (esp_monitor 32-bit): N cycles").
ESP_MONITOR_LINE = re.compile(r"^.*esp_monitor 32-bit.*$", re.MULTILINE)
BIG_INT = re.compile(r"\b(\d{4,})\b")  # FPGA cycle counts are at least 4 digits


def parse_cycles(log: Path) -> int | None:
    # logs often land as ISO-8859 (byte-level artifacts from serial capture) — errors='replace'
    text = log.read_text(encoding="utf-8", errors="replace")
    for line in ESP_MONITOR_LINE.findall(text):
        m = BIG_INT.search(line)
        if m:
            return int(m.group(1))
    return None


def split_filename(name: str) -> tuple[str, str] | None:
    """Resolve (algo, model) from a log basename.

    Accepts two naming schemes that appear in results/fpga_runs/:
      - '<algo>_<stem>'              (hand-pasted, e.g. autoencoder_opus)
      - '<stem>_<algo>_fpga_run'     (run_benchmark.sh, e.g. opus_autoencoder_fpga_run)

    Algorithm names can contain '_' (e.g. bert_tiny), so we match against the
    known sets rather than splitting by separator.
    """
    # scheme B: '<stem>_<algo>_fpga_run'
    fpga_suffix = "_fpga_run"
    if name.endswith(fpga_suffix):
        core = name[: -len(fpga_suffix)]
        for stem in MODEL_STEMS:
            prefix = f"{stem}_"
            if core.startswith(prefix):
                algo = core[len(prefix):]
                if algo in ALGORITHMS:
                    return algo, stem
    # scheme A: '<algo>_<stem>'
    for algo in ALGORITHMS:
        prefix = f"{algo}_"
        if name.startswith(prefix):
            model = name[len(prefix):]
            if model in MODEL_STEMS:
                return algo, model
    return None


def main() -> None:
    table: dict[tuple[str, str], int] = {}
    missing = []
    unparsed = []

    for log in sorted(FPGA_RUNS.glob("*.log")):
        split = split_filename(log.stem)
        if split is None:
            print(f"[skip] unrecognized filename: {log.name}")
            continue
        algo, stem = split
        cycles = parse_cycles(log)
        if cycles is None:
            unparsed.append((algo, stem))
            continue
        table[(algo, stem)] = cycles

    # full 10x5 grid, FAIL for anything without a complete cycle reading
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    out = PERF_DIR / "perf_cycles.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", *MODEL_STEMS])
        for algo in ALGORITHMS:
            row = [algo]
            for stem in MODEL_STEMS:
                v = table.get((algo, stem))
                row.append(str(v) if v is not None else "FAIL")
            w.writerow(row)

    total = len(ALGORITHMS) * len(MODEL_STEMS)
    for algo, stem in unparsed:
        missing.append((algo, stem))
    for algo in ALGORITHMS:
        for stem in MODEL_STEMS:
            if (algo, stem) not in table and (algo, stem) not in unparsed:
                missing.append((algo, stem))

    print(f"wrote {out}")
    print(f"{len(table)}/{total} parsed, "
          f"{len(unparsed)} logs present but no cycle line, "
          f"{total - len(table) - len(unparsed)} logs absent")
    for a, s in unparsed[:10]:
        print(f"  log-present-no-cycles  {a}/{s}")


if __name__ == "__main__":
    main()
