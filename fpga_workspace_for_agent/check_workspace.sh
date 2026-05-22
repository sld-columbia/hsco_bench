#!/usr/bin/env bash
# check_workspace.sh — FPGA workspace smoke-test driver
#
# Usage: ./check_workspace.sh <workspace_name> [--fix-ids]
#
# For the workspace at ./<workspace_name>/ this script:
#   0.  Parses esp_xilinx-vc707-xc7vx485t_defconfig to find which accelerators
#       are installed (each TILE line containing 'acc <UPPERCASE_NAME>'), then
#       auto-cleans this workspace's prior installs from the ESP tech tree
#       (read from .installed_accelerators.json).
#   0.3 Auto-prefixes accelerator names with the workspace name if needed
#       (via rename_accelerator.sh), then re-parses the defconfig.
#   0.5a Checks for accelerator NAME collisions against the ESP tech tree
#       and sibling workspace manifests.
#   0.5b Runs check_device_ids.py to validate that every accelerator's
#       device_id is unique within the workspace, not reserved, and not
#       already claimed by another project under $ESP_ROOT/tech/virtex7/acc/.
#       With --fix-ids, colliding/reserved IDs are auto-reassigned into
#       0x201-0x27F (modifies XML and cfg.h).
#   1+2 For each accelerator: `make bev-sim` then `make hls` in its hls/ dir.
#       bev-sim output is grep'd for 'SIMULATION FAILED' since sc_stop()
#       always exits 0 even on numerical mismatch.
#   3.  `make soft` at the workspace top level to verify baremetal compile,
#       then removes soft-build/ and socgen/.
#   4.  For each accelerator: `make install-force`, then registers the install
#       in <workspace>/.installed_accelerators.json so it can later be cleaned up
#       via `python3 install_registry.py cleanup <workspace>`.
#
# Companion tools in this directory:
#   check_device_ids.py   — device_id collision validator (used by step 0.5b)
#   install_registry.py   — manifest record / list / cleanup / cleanup-all
#   rename_accelerator.sh — accelerator renaming utility (used by step 0.3)
#
# The script sources user-provided environment script(s) that set up the
# CAD / ESP variables.  Set TOOLS_ENV to a colon-separated list of scripts
# (e.g. TOOLS_ENV="/path/a.sh:/path/b.sh").

set -eo pipefail

# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF >&2
Usage: $0 <workspace_name> [--fix-ids]
       e.g.  $0 autoencoder_vc707_opus
              $0 autoencoder_vc707_opus --fix-ids

  --fix-ids   if device-ID validation finds collisions or reserved hits,
              automatically reassign offenders into 0x201-0x27F
              (modifies XML and cfg.h files in place)

The workspace must exist as a subdirectory of:
  $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EOF
    exit 2
}

WS_NAME=""
FIX_IDS_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --fix-ids)
            FIX_IDS_FLAG="--fix-ids"
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "ERROR: unknown flag: $arg" >&2
            usage
            ;;
        *)
            if [ -z "$WS_NAME" ]; then
                WS_NAME="$arg"
            else
                echo "ERROR: unexpected extra argument: $arg" >&2
                usage
            fi
            ;;
    esac
done
[ -n "$WS_NAME" ] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$ROOT_DIR/$WS_NAME"

if [ ! -d "$WS_DIR" ]; then
    echo "ERROR: workspace not found: $WS_DIR" >&2
    exit 1
fi

DEFCONFIG="$WS_DIR/esp_xilinx-vc707-xc7vx485t_defconfig"
if [ ! -f "$DEFCONFIG" ]; then
    echo "ERROR: defconfig not found: $DEFCONFIG" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# environment setup
# ---------------------------------------------------------------------------
# TOOLS_ENV — colon-separated list of shell scripts that export the
# environment variables required by the ESP build system.
# Example:  export TOOLS_ENV="/path/to/tools_env.sh:/path/to/another_env.sh"

if [ -z "${TOOLS_ENV:-}" ]; then
    echo "ERROR: TOOLS_ENV is not set." >&2
    echo "       Export TOOLS_ENV as a colon-separated list of env scripts." >&2
    exit 1
fi

IFS=: read -ra _env_scripts <<< "$TOOLS_ENV"
for _f in "${_env_scripts[@]}"; do
    [ -z "$_f" ] && continue
    if [ ! -r "$_f" ]; then
        echo "ERROR: cannot read env script: $_f" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$_f" ""
done
unset _env_scripts _f

# Trust the env scripts — the hls/ and top-level Makefiles already contain
# $(error ...) checks for each variable they need, and will fail loudly
# with a clear message if something is missing.

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

banner() {
    echo
    echo "===================================================================="
    echo "== $*"
    echo "===================================================================="
}

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
elapsed_since() {
    local s=$1 e
    e=$(date +%s)
    printf '%dm%02ds' $(( (e-s)/60 )) $(( (e-s)%60 ))
}

# ---------------------------------------------------------------------------
# discover accelerators from defconfig  (wrapped in a function so we can
# re-parse after an auto-prefix rename)
# ---------------------------------------------------------------------------

parse_defconfig() {
    # Match lines like 'TILE_X_Y = N acc UPPERNAME dma64 0 0 sld'.
    mapfile -t ACCS_UPPER < <(
        awk '/^TILE_[0-9]+_[0-9]+[[:space:]]*=.*[[:space:]]acc[[:space:]]/ {
            for (i = 1; i <= NF; i++) if ($i == "acc") { print $(i + 1); break }
        }' "$DEFCONFIG" | sort -u
    )

    ACCS_LOWER=()
    for u in "${ACCS_UPPER[@]}"; do ACCS_LOWER+=( "${u,,}" ); done

    # Map each accelerator to the hls/ directory that hosts its Makefile.
    # Convention: <prefix>_workspace/<prefix>_sysc_catapult/hls
    HLS_DIRS=()
    for l in "${ACCS_LOWER[@]}"; do
        prefix="${l%_sysc_catapult}"
        if [ "$prefix" = "$l" ]; then
            echo "WARNING: accelerator '$l' does not end in '_sysc_catapult' — layout may differ from convention" >&2
        fi
        d="$WS_DIR/${prefix}_workspace/$l/hls"
        if [ ! -d "$d" ]; then
            echo "ERROR: cannot find HLS dir for accelerator '$l' (looked at $d)" >&2
            exit 1
        fi
        HLS_DIRS+=( "$d" )
    done
}

parse_defconfig

# ---------------------------------------------------------------------------
# plan summary
# ---------------------------------------------------------------------------

banner "Workspace check: $WS_NAME"
echo "Path:         $WS_DIR"
echo "Defconfig:    $DEFCONFIG"
echo "Accelerators: ${#ACCS_LOWER[@]}"
for i in "${!ACCS_LOWER[@]}"; do
    echo "  [$i] ${ACCS_LOWER[i]}"
    echo "       hls = ${HLS_DIRS[i]}"
done

if [ ${#ACCS_LOWER[@]} -eq 0 ]; then
    echo
    echo "NOTE: no accelerators declared in defconfig — skipping per-accelerator"
    echo "      steps. Will only run 'make soft' to check baremetal compile."
fi

TOTAL_START=$(date +%s)

# ---------------------------------------------------------------------------
# Step 0: auto-cleanup of THIS workspace's prior installs in ESP tech tree
#
# Removes anything we previously installed (recorded in
# <workspace>/.installed_accelerators.json) so the upcoming run starts
# from a clean slate. Safety floor 0x200 still applies: nothing outside
# our project's reserved range can be removed even if recorded.
#
# Uses --no-archive so we don't pile up .cleaned-*.json files on every
# check_workspace run; the new install will write a fresh manifest.
# ---------------------------------------------------------------------------

banner "pre-install cleanup of prior installs in ESP tech tree"
python3 "$ROOT_DIR/install_registry.py" cleanup "$WS_DIR" --no-archive

# ---------------------------------------------------------------------------
# Step 0.3: auto-prefix accelerator names
#
# Convention: the workspace name itself is the expected prefix.
#   bert_tiny_vc707_minimax  →  bert_tiny_vc707_minimax
#   autoencoder_vc707_kimi   →  autoencoder_vc707_kimi
#
# If an accelerator's kernel name (e.g. 'dummy') does NOT start with the
# expected prefix, it is automatically renamed (e.g. → 'autoencoder_vc707_kimi_dummy')
# using rename_accelerator.sh.  After rename, the defconfig and all source
# files are updated in-place and we re-parse.
# ---------------------------------------------------------------------------

if [[ "$WS_NAME" == *_vc707_* ]] && [ ${#ACCS_LOWER[@]} -gt 0 ]; then
    WS_PREFIX="$WS_NAME"
    NEED_REPARSE=false

    banner "auto-prefix check  (expected prefix: ${WS_PREFIX})"
    for l in "${ACCS_LOWER[@]}"; do
        kernel="${l%_sysc_catapult}"
        if [[ "$kernel" != "${WS_PREFIX}_"* ]]; then
            new_kernel="${WS_PREFIX}_${kernel}"
            echo "  rename: $kernel → $new_kernel"
            "$ROOT_DIR/rename_accelerator.sh" "$WS_DIR" "$kernel" "$new_kernel"
            NEED_REPARSE=true
        else
            echo "  ✓ $kernel already has correct prefix"
        fi
    done

    if $NEED_REPARSE; then
        echo ""
        echo "  re-parsing defconfig after rename ..."
        parse_defconfig
        echo "  accelerators now: ${ACCS_LOWER[*]}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 0.5a: accelerator NAME collision detection
#
# Two workspaces with the same accel name (e.g. both called gemm_sysc_catapult)
# would install to the same $ESP_ROOT/tech/virtex7/acc/<name>/ directory and
# silently overwrite each other's verilog.  Detect this BEFORE spending time
# on HLS.  We check every accel in our defconfig against:
#   1. Other manifests under fpga_workspace_for_agent/
#   2. Existing directories in $ESP_ROOT/tech/virtex7/acc/
# Our OWN prior installs (just cleaned in Step 0) are excluded.
# ---------------------------------------------------------------------------

banner "accelerator name collision check"
NAME_COLLISION=0
for name in "${ACCS_LOWER[@]}"; do
    tech_dir="$ESP_ROOT/tech/virtex7/acc/$name"
    if [ -d "$tech_dir" ]; then
        echo "COLLISION: '$name' already exists at $tech_dir" >&2
        echo "           (installed by another workspace — NOT ours, we just cleaned)" >&2
        NAME_COLLISION=1
    fi
    # Also scan sibling manifests for the same name
    for manifest in "$ROOT_DIR"/*/.installed_accelerators.json; do
        [ -f "$manifest" ] || continue
        manifest_ws="$(dirname "$manifest")"
        [ "$manifest_ws" = "$WS_DIR" ] && continue   # skip our own
        if python3 -c "
import json, sys
data = json.load(open('$manifest'))
names = [e['name'] for e in data.get('installed', [])]
sys.exit(0 if '$name' in names else 1)
" 2>/dev/null; then
            echo "COLLISION: '$name' also recorded in manifest of $(basename "$manifest_ws")" >&2
            NAME_COLLISION=1
        fi
    done
done
if [ "$NAME_COLLISION" -ne 0 ]; then
    echo "" >&2
    echo "ERROR: accelerator name collision(s) detected — aborting." >&2
    echo "Fix: rename your accelerator with:" >&2
    echo "  ./rename_accelerator.sh $(basename "$WS_DIR") <old_name> <new_name>" >&2
    exit 1
fi
echo "  ✓ no name collisions"

# ---------------------------------------------------------------------------
# Step 0.5b: device-ID validation (fail fast before HLS)
# ---------------------------------------------------------------------------

banner "device-ID validation${FIX_IDS_FLAG:+ (with --fix-ids)}"
python3 "$ROOT_DIR/check_device_ids.py" "$WS_DIR" $FIX_IDS_FLAG

# ---------------------------------------------------------------------------
# Step 1 + 2: bev-sim and hls per accelerator
# ---------------------------------------------------------------------------

for i in "${!ACCS_LOWER[@]}"; do
    name="${ACCS_LOWER[i]}"
    dir="${HLS_DIRS[i]}"

    banner "[$name] make bev-sim   (start $(timestamp))"
    t0=$(date +%s)
    # Use a subshell with cd (not 'make -C'): the workspace Makefiles use
    # $(PWD) from the shell env, which 'make -C' does not update.
    #
    # bev-sim's sim_sc binary calls sc_stop() which ALWAYS exits 0, even
    # when the testbench printed "SIMULATION FAILED" due to a numerical
    # mismatch. Capture stdout+stderr via tee and grep the captured log
    # to actually detect those failures.
    bev_log="$dir/bev-sim.log"
    ( cd "$dir" && make bev-sim-clean && make bev-sim ) 2>&1 | tee "$bev_log"
    if grep -q 'SIMULATION FAILED' "$bev_log"; then
        echo "" >&2
        echo "ERROR: [$name] testbench reported SIMULATION FAILED" >&2
        echo "       (full output preserved at $bev_log)" >&2
        exit 1
    fi
    rm -f "$bev_log"
    echo "[$name] bev-sim OK  (elapsed $(elapsed_since $t0))"

    banner "[$name] make hls   (start $(timestamp))"
    t0=$(date +%s)
    ( cd "$dir" && make hls-clean && make hls )
    echo "[$name] hls OK  (elapsed $(elapsed_since $t0))"
done

# ---------------------------------------------------------------------------
# Step 3: make soft (compile check) + cleanup
# ---------------------------------------------------------------------------

banner "make soft   (start $(timestamp))"
t0=$(date +%s)
( cd "$WS_DIR" && make soft )
echo "soft OK  (elapsed $(elapsed_since $t0))"

banner "cleanup: rm -rf soft-build socgen"
rm -rf "$WS_DIR/soft-build" "$WS_DIR/socgen"
echo "cleaned"

# ---------------------------------------------------------------------------
# Step 4: make install per accelerator
# ---------------------------------------------------------------------------

for i in "${!ACCS_LOWER[@]}"; do
    name="${ACCS_LOWER[i]}"
    dir="${HLS_DIRS[i]}"

    banner "[$name] make install-force   (start $(timestamp))"
    t0=$(date +%s)
    # 'install-force' is the non-interactive variant of 'install' in the
    # hls/Makefile (removes any prior install without prompting).
    ( cd "$dir" && make install-force )
    # Record the install in this workspace's manifest so we can later run
    # `install_registry.py cleanup <workspace>` to remove just our installs
    # from the shared $ESP_ROOT/tech/virtex7/acc/ tree.
    if ! python3 "$ROOT_DIR/install_registry.py" record "$WS_DIR" "$name"; then
        echo "WARNING: failed to record install of $name in registry" >&2
    fi
    echo "[$name] install-force OK  (elapsed $(elapsed_since $t0))"
done

banner "ALL CHECKS PASSED — $WS_NAME   (total $(elapsed_since $TOTAL_START))"
