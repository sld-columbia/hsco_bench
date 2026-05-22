#!/usr/bin/env bash
# run_benchmark.sh — FPGA accelerator benchmark campaign driver.
#
# Iterates each LLM across every algorithm: reset workspace → run agent →
# check_workspace → (optional) FPGA build/run → cleanup. One iteration
# failure is logged and the campaign continues.
#
# Flags:
#   --dry-run     print the commands that would run, without executing
#   --skip-fpga   run agent + check_workspace only (skip vivado-syn / fpga-run)
#   --build-only  run through vivado-syn (bitstream) but skip programming / fpga-run
#   -h | --help
#
# Output:
#   results/summary.csv                      — campaign summary
#   results/logs/<model>_<algo>.log          — per-iteration log
#   results/fpga_runs/<model>_<algo>_fpga_run.log  — raw fpga-run output
#   trajectories/<model>_<algo>.json         — mini-swe-agent trajectory

set -uo pipefail   # no -e at top level — per-iteration failures must not abort the campaign

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="${TEMPLATE_DIR:-$(dirname "$ROOT_DIR")/hsco_testcases}"
RESULTS_DIR="$ROOT_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"
FPGA_LOG_DIR="$RESULTS_DIR/fpga_runs"
TRAJ_DIR="$ROOT_DIR/trajectories"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
CAMPAIGN_LOG="$RESULTS_DIR/campaign.log"  # one-line-per-event campaign digest

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# User-listed algorithm set. Only those with an existing template dir will
# run; the rest are logged as skipped (NO_TEMPLATE) and the campaign moves on.
ALGORITHMS=(
    autoencoder
    bert_tiny
    efficientvit
    minideit
    minilm
    mobilenetv4
    tinyclip_text
    tinyclip_vision_single
    nightvision
    wami
)

# Model stem → filename of mini-docker-<stem>.yaml, plus human-readable label.
MODEL_STEMS=(minimax kimi opus gpt gemini)
declare -A MODEL_LABEL=(
    [minimax]=minimax-m2.5
    [kimi]=kimi-k2.5
    [opus]=opus-4.6
    [gpt]=gpt-5.4
    [gemini]=gemini-3.1-pro
)

# DSU identity — workspace must match these or the iteration is skipped.
EXPECTED_DSU_IP="C0A80107"
EXPECTED_DSU_ETH="A6A7A0F8043D"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
SKIP_FPGA=0
BUILD_ONLY=0
MODELS_FILTER=""
ALGOS_FILTER=""

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--skip-fpga|--build-only] [--models LIST] [--algos LIST] [-h|--help]

  --dry-run       echo commands without executing them
  --skip-fpga     run agent + check_workspace only (skip vivado-syn / fpga-run)
  --build-only    run through vivado-syn (bitstream) but skip programming / fpga-run
  --models LIST   comma-separated subset of model stems (e.g. 'kimi' or 'kimi,gpt')
  --algos  LIST   comma-separated subset of algorithms   (e.g. 'autoencoder,bert_tiny')
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)    DRY_RUN=1 ;;
        --skip-fpga)  SKIP_FPGA=1 ;;
        --build-only) BUILD_ONLY=1 ;;
        --models)     MODELS_FILTER="$2"; shift ;;
        --models=*)   MODELS_FILTER="${1#*=}" ;;
        --algos)      ALGOS_FILTER="$2"; shift ;;
        --algos=*)    ALGOS_FILTER="${1#*=}" ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "ERROR: unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [ "$SKIP_FPGA" -eq 1 ] && [ "$BUILD_ONLY" -eq 1 ]; then
    echo "ERROR: --skip-fpga and --build-only are mutually exclusive" >&2
    exit 2
fi

# Apply --models / --algos filters against the canonical lists.
apply_filter() {
    # apply_filter "<csv>" <canonical_array_name> <out_array_name>
    local csv="$1" src_name="$2" out_name="$3"
    local -n src_ref="$src_name"
    local -n out_ref="$out_name"
    if [ -z "$csv" ]; then
        out_ref=( "${src_ref[@]}" )
        return 0
    fi
    IFS=',' read -ra want <<<"$csv"
    out_ref=()
    for w in "${want[@]}"; do
        w="${w// /}"  # strip spaces
        local found=0
        for s in "${src_ref[@]}"; do
            if [ "$s" = "$w" ]; then out_ref+=( "$s" ); found=1; break; fi
        done
        if [ "$found" -eq 0 ]; then
            echo "ERROR: '$w' not in $src_name: ${src_ref[*]}" >&2
            exit 2
        fi
    done
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
elapsed()   { local t=$1; echo $(( $(date +%s) - t )); }

# log LEVEL "message"
#   Always prints to stdout. Also appends to $CAMPAIGN_LOG unless --dry-run,
#   so previewing the campaign never mutates persistent artifacts.
#   LEVEL ∈ {INFO, WARN, ERR}. Verbose tool output still goes to the
#   per-iteration log ($iter_log) — this helper is for campaign-level state.
log() {
    local lvl="$1"; shift
    local line
    line=$(printf '%s [%-4s] %s' "$(timestamp)" "$lvl" "$*")
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s\n' "$line"
    else
        printf '%s\n' "$line" | tee -a "$CAMPAIGN_LOG"
    fi
}

banner() {
    echo
    echo "===================================================================="
    echo "== $(timestamp)  $*"
    echo "===================================================================="
}

# echo + run (suppressed under --dry-run)
run() {
    printf '[run] %s\n' "$*"
    if [ "$DRY_RUN" -eq 0 ]; then
        eval "$@"
    fi
}

activate_conda() {
    [ "$DRY_RUN" -eq 1 ] && return 0
    if [ -z "${_CONDA_ACTIVATED:-}" ]; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
        # shellcheck disable=SC1091
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate mini-swe-agent
        export _CONDA_ACTIVATED=1
    fi
}

# Source user-provided environment script(s) for CAD / ESP tools.
# TOOLS_ENV is a colon-separated list of scripts.
activate_cad_env() {
    [ "$DRY_RUN" -eq 1 ] && return 0
    if [ -z "${_CAD_ENV_ACTIVATED:-}" ]; then
        if [ -z "${TOOLS_ENV:-}" ]; then
            echo "ERROR: TOOLS_ENV is not set." >&2
            exit 1
        fi
        local IFS=:
        local _scripts
        read -ra _scripts <<< "$TOOLS_ENV"
        for _f in "${_scripts[@]}"; do
            [ -z "$_f" ] && continue
            # shellcheck disable=SC1090
            source "$_f" ""
        done
        export _CAD_ENV_ACTIVATED=1
    fi
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$FPGA_LOG_DIR" "$TRAJ_DIR"
touch "$CAMPAIGN_LOG"

if [ ! -f "$SUMMARY_CSV" ]; then
    echo "model,algorithm,agent_time_s,check_pass,synth_pass,fpga_run_pass,fpga_run_log_path" > "$SUMMARY_CSV"
fi

append_csv() {
    # append_csv <model> <algo> <agent_s> <check> <synth> <fpga> <log_path>
    # Skipped under --dry-run so previews don't pile DRY rows into the CSV.
    [ "$DRY_RUN" -eq 1 ] && return 0
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$@" >> "$SUMMARY_CSV"
}

# ---------------------------------------------------------------------------
# Per-iteration work
# ---------------------------------------------------------------------------
run_iteration() {
    local model_stem="$1" algo="$2"
    local model_label="${MODEL_LABEL[$model_stem]}"
    # Workspace copy is named per (algo, model_stem) so parallel runs from
    # different LLMs don't clobber each other. Template always lives under
    # <algo>_vc707_clear_testcase — its name is fixed.
    local ws_name="${algo}_vc707_${model_stem}"
    local ws_path="$ROOT_DIR/$ws_name"
    local tmpl_path="$TEMPLATE_DIR/${algo}_vc707_clear_testcase"
    local yaml_path="$ROOT_DIR/mini-docker-${model_stem}.yaml"
    local traj_path="$TRAJ_DIR/${model_stem}_${algo}.json"
    local iter_log="$LOG_DIR/${model_stem}_${algo}.log"
    local fpga_log="$FPGA_LOG_DIR/${model_stem}_${algo}_fpga_run.log"
    local agent_time=0 check_pass=SKIP synth_pass=SKIP fpga_pass=SKIP fpga_log_out=""

    banner "[$model_label / $algo]"
    : > "$iter_log"
    log INFO "$model_label | $algo | begin  (log: $iter_log)"
    local iter_start
    iter_start=$(date +%s)

    # --- 0. sanity: template + yaml must exist --------------------------
    if [ ! -d "$tmpl_path" ]; then
        log WARN "$model_label | $algo | skip: template missing ($tmpl_path)"
        append_csv "$model_label" "$algo" 0 NO_TEMPLATE SKIP SKIP ""
        return 0
    fi
    if [ ! -f "$yaml_path" ]; then
        log WARN "$model_label | $algo | skip: model yaml missing ($yaml_path)"
        append_csv "$model_label" "$algo" 0 NO_YAML SKIP SKIP ""
        return 0
    fi

    # --- 0.5 DSU identity check -------------------
    local defconfig="$tmpl_path/esp_xilinx-vc707-xc7vx485t_defconfig"
    local ip eth
    ip=$(awk -F'= *' '/^CONFIG_DSU_IP[[:space:]]*=/{print $2; exit}'  "$defconfig" | tr -d '[:space:]')
    eth=$(awk -F'= *' '/^CONFIG_DSU_ETH[[:space:]]*=/{print $2; exit}' "$defconfig" | tr -d '[:space:]')
    if [ "$ip" != "$EXPECTED_DSU_IP" ] || [ "$eth" != "$EXPECTED_DSU_ETH" ]; then
        log ERR "$model_label | $algo | skip: DSU mismatch (ip=$ip eth=$eth)"
        append_csv "$model_label" "$algo" 0 DSU_MISMATCH SKIP SKIP ""
        return 0
    fi

    # --- 1. reset workspace from template -------------------------------
    run "rm -rf '$ws_path'"
    run "cp -r '$tmpl_path' '$ws_path'"

    # --- 2. patch yaml: rewrite the "-v HOST:/workspace" bind mount -----
    # Replace any path before ":/workspace" with ROOT_DIR/ws_name. Handles both
    # the __ROOT_DIR__ placeholder and previously-patched absolute paths.
    run "sed -i -E 's|[^\"]*:/workspace|${ROOT_DIR}/${ws_name}:/workspace|' '$yaml_path'"

    # --- 3. run the agent ------------------------------------------------
    activate_conda
    local t0
    t0=$(date +%s)
    # --yolo              — auto-accept all action confirmations (the CLI form of /y)
    # --exit-immediately  — skip the interactive "press Enter to exit" prompt at end
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[run] (cd $ws_path && mini -y --exit-immediately -c mini.yaml -c ../mini-docker-${model_stem}.yaml -t '...' -o $traj_path)"
    else
        (
            cd "$ws_path" && \
            mini --yolo --exit-immediately \
                 -c mini.yaml \
                 -c "../mini-docker-${model_stem}.yaml" \
                 -t "Please take a look at the README.md file and finish the job." \
                 -o "$traj_path"
        ) >> "$iter_log" 2>&1 || echo "WARN: agent step exited non-zero" >> "$iter_log"
    fi
    agent_time=$(elapsed "$t0")
    log INFO "$model_label | $algo | agent done (${agent_time}s)"

    # --- 4. check_workspace ----------------------------------------------
    # --fix-ids auto-reassigns device-ID collisions against stale ESP tech-tree
    # installs (e.g. 0x201 leftover from a prior campaign). Without it every
    # iteration following the README's "start at 0x201" guidance collides.
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[run] ./check_workspace.sh $ws_name --fix-ids"
        check_pass=DRY
    else
        if ( cd "$ROOT_DIR" && ./check_workspace.sh "$ws_name" --fix-ids ) >> "$iter_log" 2>&1; then
            check_pass=PASS
        else
            check_pass=FAIL
        fi
    fi
    log INFO "$model_label | $algo | check=$check_pass"

    # --- 5. FPGA build + run (only if check passed and not --skip-fpga) --
    # --build-only stops after vivado-syn (bitstream produced, board untouched).
    if [ "$SKIP_FPGA" -eq 1 ]; then
        log INFO "$model_label | $algo | fpga skipped (--skip-fpga)"
    elif [ "$check_pass" != "PASS" ]; then
        log INFO "$model_label | $algo | fpga skipped (check did not pass)"
    elif [ "$DRY_RUN" -eq 1 ]; then
        if [ "$BUILD_ONLY" -eq 1 ]; then
            echo "[run] (cd $ws_path && make esp-config && make vivado-syn)"
        else
            echo "[run] (cd $ws_path && make esp-config && make vivado-syn && make fpga-program && rm -rf soft-build && make soft && ./load_model_ariane.sh or load_data_ariane.sh && make fpga-run)"
        fi
        synth_pass=DRY; fpga_pass=DRY
    else
        activate_cad_env
        local fb0
        fb0=$(date +%s)

        # Synthesis gate — esp-config + vivado-syn must both succeed.
        if ( cd "$ws_path" && make esp-config && make vivado-syn ) >> "$iter_log" 2>&1; then
            synth_pass=PASS
        else
            synth_pass=FAIL
        fi

        if [ "$synth_pass" = "PASS" ] && [ "$BUILD_ONLY" -eq 0 ]; then
            (
                cd "$ws_path" && \
                make fpga-program && \
                rm -rf soft-build && \
                make soft &&
                # Some testcases use load_data_ariane.sh instead of load_model_ariane.sh
                # (e.g. nightvision, wami, tinyclip_text, tinyclip_vision_single).
                if [ -x ./load_model_ariane.sh ]; then ./load_model_ariane.sh; else ./load_data_ariane.sh; fi
            ) >> "$iter_log" 2>&1 || echo "WARN: fpga-program / soft / load step failed" >> "$iter_log"

            # fpga-run: capture full output to a dedicated log. sc_stop()-style
            # tests often exit 0 even on numerical failure, so we combine exit
            # code with a grep for explicit FAIL markers.
            ( cd "$ws_path" && make fpga-run ) > "$fpga_log" 2>&1
            local rc=$?
            cat "$fpga_log" >> "$iter_log"
            fpga_log_out="$fpga_log"
            if [ $rc -eq 0 ] && ! grep -q -E -i 'FAIL|MISMATCH|ERROR' "$fpga_log"; then
                fpga_pass=PASS
            else
                fpga_pass=FAIL
            fi
        fi

        if [ "$BUILD_ONLY" -eq 1 ]; then
            log INFO "$model_label | $algo | synth=$synth_pass run=SKIP (--build-only, fpga $(elapsed "$fb0")s)"
        else
            log INFO "$model_label | $algo | synth=$synth_pass run=$fpga_pass (fpga $(elapsed "$fb0")s)"
        fi
    fi

    # Cleanup steps intentionally omitted — caller handles disk / ESP-tree
    # hygiene manually between campaigns.

    append_csv "$model_label" "$algo" "$agent_time" "$check_pass" "$synth_pass" "$fpga_pass" "$fpga_log_out"
    log INFO "$model_label | $algo | done  ($(elapsed "$iter_start")s total)"
}

# ---------------------------------------------------------------------------
# Main loop — one model at a time, all algorithms, then next model
# ---------------------------------------------------------------------------
CAMPAIGN_START=$(date +%s)
declare -a ACTIVE_MODELS ACTIVE_ALGOS
apply_filter "$MODELS_FILTER" MODEL_STEMS ACTIVE_MODELS
apply_filter "$ALGOS_FILTER"  ALGORITHMS  ACTIVE_ALGOS

log INFO "=== campaign start  (dry_run=$DRY_RUN skip_fpga=$SKIP_FPGA build_only=$BUILD_ONLY) ==="
log INFO "models: ${ACTIVE_MODELS[*]}"
log INFO "algos : ${ACTIVE_ALGOS[*]}"

for model_stem in "${ACTIVE_MODELS[@]}"; do
    log INFO "--- model: ${MODEL_LABEL[$model_stem]} (stem=$model_stem) ---"
    banner "MODEL: ${MODEL_LABEL[$model_stem]}  (stem=$model_stem)"
    for algo in "${ACTIVE_ALGOS[@]}"; do
        # Contain iteration failures so one bad run does not abort the campaign.
        run_iteration "$model_stem" "$algo" || \
            log WARN "$model_stem | $algo | run_iteration returned non-zero"
    done
done

log INFO "=== campaign done ($(elapsed "$CAMPAIGN_START")s) — summary: $SUMMARY_CSV ==="
