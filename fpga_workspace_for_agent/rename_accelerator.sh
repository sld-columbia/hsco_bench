#!/usr/bin/env bash
set -euo pipefail
# rename_accelerator.sh — safely rename an accelerator across an ESP workspace
#
# Usage:
#   ./rename_accelerator.sh <workspace> <old_prefix> <new_prefix> [--dry-run]
#
# Examples:
#   ./rename_accelerator.sh bert_tiny_vc707_minimax gemm minimax_gemm --dry-run
#   ./rename_accelerator.sh bert_tiny_vc707_minimax gemm minimax_gemm
#
# What it does (in order):
#   1. sed all text-file CONTENTS:  old → new  and  OLD → NEW  (case-preserving)
#   2. Rename FILES whose names contain the old prefix
#   3. Rename DIRECTORIES whose names contain the old prefix (innermost first)
#
# Safety:
#   - dummy_workspace / dummy_sysc_catapult are always excluded
#   - Symlinks are NOT followed
#   - Only known text extensions are modified (*.h, *.c, *.cpp, *.hpp, *.xml,
#     *.tcl, *.json, *.md, *.sh, *.py, *.txt, Makefile, *defconfig*)
#   - Run with --dry-run first to preview every change

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: $0 <workspace> <old_prefix> <new_prefix> [--dry-run]" >&2
    exit 1
}
[ $# -lt 3 ] && usage

WS_ARG="$1"; OLD="$2"; NEW="$3"
DRY_RUN=false
[ "${4:-}" = "--dry-run" ] && DRY_RUN=true

[ "$OLD" = "$NEW" ] && { echo "ERROR: old and new names are identical" >&2; exit 1; }

# Resolve workspace path (relative to script dir, or absolute)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/$WS_ARG" ]; then
    WS_DIR="$SCRIPT_DIR/$WS_ARG"
elif [ -d "$WS_ARG" ]; then
    WS_DIR="$(cd "$WS_ARG" && pwd)"
else
    echo "ERROR: workspace '$WS_ARG' not found" >&2; exit 1
fi

OLD_U=$(echo "$OLD" | tr 'a-z' 'A-Z')
NEW_U=$(echo "$NEW" | tr 'a-z' 'A-Z')

echo "============================================================"
echo " Rename accelerator: $OLD → $NEW"
echo " Workspace:          $(basename "$WS_DIR")"
echo " UPPERCASE:          $OLD_U → $NEW_U"
$DRY_RUN && echo " MODE:               DRY RUN (no changes)"
echo "============================================================"
echo ""

# Quick sanity: does the old name actually exist in the workspace?
HIT=$(find "$WS_DIR" -maxdepth 3 -name "*${OLD}*" \
    -not -path '*/dummy_workspace/*' -not -path '*/dummy_sysc_catapult/*' | head -1)
if [ -z "$HIT" ]; then
    echo "ERROR: no file or directory containing '$OLD' found." >&2
    echo "       Double-check the old prefix." >&2
    exit 1
fi

# Check that new name doesn't already exist (would collide)
for candidate in "${NEW}_workspace" "${NEW}_sw_driver" "${NEW}_sysc_catapult"; do
    COLLISION=$(find "$WS_DIR" -maxdepth 3 -name "$candidate" -type d 2>/dev/null | head -1)
    if [ -n "$COLLISION" ]; then
        echo "ERROR: '$candidate' already exists — rename would collide" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Replace strings inside file contents
# ---------------------------------------------------------------------------
echo "--- Step 1: string replacement inside files ---"
count=0
while IFS= read -r -d '' f; do
    if grep -qlE "${OLD}|${OLD_U}" "$f" 2>/dev/null; then
        rel="${f#$WS_DIR/}"
        echo "  [sed] $rel"
        if ! $DRY_RUN; then
            # UPPERCASE first, then lowercase — avoids partial-match issues
            # when OLD is a substring of NEW (e.g. gemm → minimax_gemm).
            sed -i "s/${OLD_U}/${NEW_U}/g; s/${OLD}/${NEW}/g" "$f"
        fi
        count=$((count + 1))
    fi
done < <(find "$WS_DIR" -maxdepth 6 \
    \( -name '*.h' -o -name '*.c' -o -name '*.cpp' -o -name '*.hpp' \
       -o -name '*.xml' -o -name '*.tcl' -o -name '*.txt' -o -name '*.json' \
       -o -name '*.md' -o -name '*.sh' -o -name '*.py' \
       -o -name 'Makefile' -o -name '*defconfig*' \) \
    -not -path '*/dummy_workspace/*' \
    -not -path '*/dummy_sysc_catapult/*' \
    -not -name '*.log' \
    -type f -print0 2>/dev/null)
echo "  → $count file(s)"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Rename files whose names contain the old prefix
# ---------------------------------------------------------------------------
echo "--- Step 2: rename files ---"
while IFS= read -r f; do
    d=$(dirname "$f"); b=$(basename "$f")
    nb=$(echo "$b" | sed "s/${OLD}/${NEW}/g")
    rel="${f#$WS_DIR/}"
    echo "  [mv] $rel → $(dirname "$rel")/$nb"
    $DRY_RUN || mv "$f" "$d/$nb"
done < <(find "$WS_DIR" -maxdepth 6 -depth -name "*${OLD}*" -type f \
    -not -path '*/dummy_workspace/*' -not -path '*/dummy_sysc_catapult/*' 2>/dev/null)
echo ""

# ---------------------------------------------------------------------------
# Step 3: Rename directories (innermost first via -depth)
# ---------------------------------------------------------------------------
echo "--- Step 3: rename directories ---"
while IFS= read -r d; do
    parent=$(dirname "$d"); b=$(basename "$d")
    nb=$(echo "$b" | sed "s/${OLD}/${NEW}/g")
    rel="${d#$WS_DIR/}"
    echo "  [mv] $rel → $(dirname "$rel")/$nb"
    $DRY_RUN || mv "$d" "$parent/$nb"
done < <(find "$WS_DIR" -maxdepth 6 -depth -name "*${OLD}*" -type d \
    -not -path '*/dummy_workspace/*' -not -path '*/dummy_sysc_catapult/*' 2>/dev/null)
echo ""

# ---------------------------------------------------------------------------
# Post-rename verification (skip in dry-run — nothing actually changed)
# ---------------------------------------------------------------------------
if ! $DRY_RUN; then
    echo "--- Post-rename verification ---"
    # Use word-boundary \b to avoid false positives when OLD is a
    # substring of NEW (e.g. 'gemm' inside 'bert_tiny_minimax_gemm').
    # \b requires a non-word char (or edge) on both sides, so
    # '_gemm_' won't match but a standalone 'gemm' will.
    leftover=$(grep -rlP "\b${OLD}\b|\b${OLD_U}\b" "$WS_DIR" \
        --include='*.h' --include='*.c' --include='*.cpp' --include='*.hpp' \
        --include='*.xml' --include='*.tcl' --include='Makefile' \
        --include='*defconfig*' --include='*.json' --include='*.md' \
        2>/dev/null \
        | grep -v 'dummy_workspace\|dummy_sysc_catapult' \
        | head -10 || true)
    if [ -n "$leftover" ]; then
        echo "  WARNING: '$OLD' / '$OLD_U' still found in:"
        echo "$leftover" | while IFS= read -r l; do echo "    ${l#$WS_DIR/}"; done
        echo "  (may be in comments or unrelated — please verify)"
    else
        echo "  ✓ no remaining references to '$OLD' / '$OLD_U'"
    fi

    # Warn about stale tech-tree install
    if [ -n "${ESP_ROOT:-}" ]; then
        tech_old="$ESP_ROOT/tech/virtex7/acc/${OLD}_sysc_catapult"
        if [ -d "$tech_old" ]; then
            echo ""
            echo "  WARNING: old accelerator still in tech tree:"
            echo "    $tech_old"
            echo "  Run:  python3 install_registry.py cleanup $(basename "$WS_DIR")"
            echo "  Then: ./check_workspace.sh $(basename "$WS_DIR")"
        fi
    fi
fi

echo ""
echo "Done.$(${DRY_RUN} && echo ' (dry run — nothing was actually changed)')"
