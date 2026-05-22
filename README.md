# HSCO Benchmark

SoC benchmark system for evaluating LLM-driven hardware-software co-design.

## Directory Structure

```
hsco_bench/
├── fpga_workspace_for_agent/       # Benchmark driver scripts + model configs
│   ├── run_benchmark.sh            # Main campaign driver
│   ├── check_workspace.sh          # Workspace validation + HLS build + install
│   ├── rename_accelerator.sh       # Accelerator renaming utility
│   ├── install_registry.py         # ESP tech-tree manifest management
│   ├── check_device_ids.py         # Device ID collision validator
│   ├── mini-docker-*.yaml          # Per-model agent configurations
│   └── results/                    # Result extraction scripts
│       ├── extract_area_matrix.py      # Parse Vivado utilization reports → area CSV
│       ├── extract_cost_matrix.py      # Extract API cost from trajectories → cost CSV
│       ├── extract_perf_matrix.py      # Parse FPGA run logs → performance CSV
│       └── area/area_compile/
│           └── recompute_area.py       # Recompute area with 3-resource formula (logic cells + BRAM + DSP)
├── hsco_testcases/                 # Template workspaces (one per algorithm)
├── LICENSE
└── README.md
```

## Prerequisites

This benchmark requires the [ESP (Embedded Scalable Platforms)](https://www.esp.cs.columbia.edu/) framework.  
For installation instructions, see: https://www.esp.cs.columbia.edu/docs/setup/

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ESP_ROOT` | Path to ESP installation root | `/opt/esp` |
| `CATAPULT_HOME` | Catapult HLS `Mgc_home` directory | `/opt/Catapult/Mgc_home` |
| `MATCHLIB_HOME` | MatchLib toolkit root (containing `include/`) | `/opt/matchlib_toolkit` |
| `TOOLS_ENV` | Colon-separated list of shell scripts that export the CAD / ESP environment variables (e.g. `PATH`, tool homes) | `/path/to/tools_env.sh` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEMPLATE_DIR` | `../hsco_testcases` (relative to script) | Template workspace directory |

### API Keys

The benchmark invokes multiple LLM providers. Set the corresponding API keys:

| Provider | Environment Variable |
|----------|---------------------|
| Anthropic (Opus) | `ANTHROPIC_API_KEY` |
| OpenAI (GPT) | `OPENAI_API_KEY` |
| Google (Gemini) | `GEMINI_API_KEY` |
| DeepInfra (MiniMax, Kimi) | `DEEPINFRA_API_KEY` |

### Software Dependencies

- **ESP toolchain**: Xilinx Vivado, Catapult HLS, RISC-V cross-compiler
- **[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent/)**: LLM agent framework (install into a `mini-swe-agent` conda environment)
- **Docker**: Required for sandboxed agent execution
- **Python 3.8+**: For `install_registry.py` and `check_device_ids.py`

## Usage

```bash
# Set required environment
export ESP_ROOT=/path/to/esp

# Run full campaign
cd fpga_workspace_for_agent
./run_benchmark.sh

# Dry run (preview commands without executing)
./run_benchmark.sh --dry-run

# Run specific models/algorithms only
./run_benchmark.sh --models opus,gpt --algos bert_tiny,minideit

# Build bitstream only (skip FPGA programming)
./run_benchmark.sh --build-only

# Skip FPGA steps entirely (agent + check_workspace only)
./run_benchmark.sh --skip-fpga
```

## Output

Raw results are written to `fpga_workspace_for_agent/results/`:

- `summary.csv` — One row per (model, algorithm) with pass/fail status
- `logs/<model>_<algo>.log` — Full per-iteration log
- `fpga_runs/<model>_<algo>_fpga_run.log` — Raw FPGA run output
- `campaign.log` — Campaign-level event digest

### Extracting Metrics

After a campaign completes, run the extraction scripts to produce structured CSVs:

```bash
cd fpga_workspace_for_agent/results
python extract_cost_matrix.py      # → cost/cost_usd.csv
python extract_perf_matrix.py      # → perf/perf_cycles.csv
python extract_area_matrix.py      # → area/area_avg_util.csv (+ area_luts_pct, area_ffs_pct, area_dsps_pct, area_bram_pct)
```

## Citation

If you find our work helpful, please use the following citation.

```bibtex
@misc{tsai2026hscobenchagentdrivenendtoendhardwaresoftware,
      title={HSCO-Bench: An Agent-Driven End-to-End Hardware-Software Co-design Benchmark for Systems-on-Chip}, 
      author={Pei-Huan Tsai and Kuan-Lin Chiu and William Baisi and Pin-Yu Chen and Luca P. Carloni},
      year={2026},
      eprint={2605.19399},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2605.19399}, 
}
```

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Some files derived from the [ESP](https://www.esp.cs.columbia.edu/) framework retain their original Apache 2.0 headers.