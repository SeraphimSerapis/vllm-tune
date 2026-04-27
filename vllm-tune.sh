#!/bin/bash
set -euo pipefail
# vllm-tune — Unified kernel tuning CLI for vLLM
#
# Consolidates MoE and FP8 dense GEMM kernel tuning into a single command.
# Tunes kernels inside a running spark-vllm-docker container, stores configs
# locally, and can deploy them via docker cp (no mod required).
#
# Tuning runs in tmux by default so long sessions survive SSH disconnects.
#
# Usage:
#   vllm-tune.sh <MODEL_ID> [options]
#   vllm-tune.sh --attach [MODEL_ID]     Attach to a running tuning session
#
# Options:
#   --tp <N>              Tensor parallelism (default: 2)
#   --mode <MODE>         moe | fp8 | all (default: all)
#   --batch-size <S...>   Custom batch sizes
#   --shapes <N,K ...>    Explicit FP8 shapes (skip auto-detect)
#   --dtype <DTYPE>       MoE dtype (default: fp8_w8a8)
#   -t, --target <NAME>   Container name (default: vllm_node)
#   --deploy              Deploy configs to container after tuning
#   --deploy-only         Skip tuning, just deploy existing configs
#   --attach              Attach to existing tmux tuning session
#   --foreground          Run in foreground instead of tmux
#   --sync-mod            Sync configs back to vllm-tune mod dir
#   --mod-dir <DIR>        Override mod directory (default: auto-detected)
#   --dry-run             Show plan without executing
#   -h, --help            Show this help
#
# Examples:
#   vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy
#   vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --mode moe
#   vllm-tune.sh --attach                     # reattach to session
#   vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --deploy-only -t vllm_node
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TUNE_SCRIPTS_DIR="${TUNE_SCRIPTS_DIR:-$SCRIPT_DIR}"
VERSION="0.1.3"

# Source shared library (config helpers, etc.)
source "$SCRIPT_DIR/lib/common.sh"

# Read paths and defaults from config.json, with env-var and hardcoded fallbacks
CONFIG_HOME="${VLLM_TUNE_HOME:-$(cfg_get configs_dir)}"
CONFIG_HOME="${CONFIG_HOME:-$SCRIPT_DIR/configs}"

# Container target paths for vLLM kernel configs
# Auto-detected at deploy time via detect_vllm_paths()
VLLM_MOE_PATH=""
VLLM_FP8_PATH=""

# Resolve the vLLM install path inside the container (handles any Python version).
detect_vllm_paths() {
    local vllm_root
    vllm_root=$(docker exec "$CONTAINER" python3 -c \
        "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)" 2>/dev/null) \
        || vllm_root="/usr/local/lib/python3.12/dist-packages/vllm"
    VLLM_MOE_PATH="$vllm_root/model_executor/layers/fused_moe/configs"
    VLLM_FP8_PATH="$vllm_root/model_executor/layers/quantization/utils/configs"
}

# Defaults — config.json values override hardcoded defaults, CLI flags override both
MODEL=""
TP=$(cfg_get tp); TP="${TP:-2}"
MODE="all"
DTYPE=$(cfg_get dtype); DTYPE="${DTYPE:-fp8_w8a8}"
CONTAINER=$(cfg_get container); CONTAINER="${CONTAINER:-vllm_node}"
BATCH_SIZES=()
SHAPES=()
DO_DEPLOY=false
DEPLOY_ONLY=false
USE_TMUX=true
ATTACH_ONLY=false
SYNC_MOD=false
FORCE_SETUP=false
MOD_DIR_OVERRIDE=""
DRY_RUN=false

# ── Helpers ──────────────────────────────────────────────────────────

die()  { echo "Error: $*" >&2; exit 1; }
info() { printf "  \033[1;34mℹ\033[0m %s\n" "$*"; }
ok()   { printf "  \033[1;32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[1;33m⚠\033[0m %s\n" "$*" >&2; }

model_slug() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's|/|--|g; s/[^a-z0-9._-]/-/g'
}

show_help() {
    sed -n '/^# vllm-tune/,/^$/{s/^# \?//;p}' "$0"
    exit 0
}

show_banner() {
    printf "\033[1;36m"
    cat <<'EOF'

██    ██ ██      ██      ███    ███       ████████ ██    ██ ███    ██ ███████
██    ██ ██      ██      ████  ████          ██    ██    ██ ████   ██ ██
██    ██ ██      ██      ██ ████ ██ ███████  ██    ██    ██ ██ ██  ██ █████
 ██  ██  ██      ██      ██  ██  ██          ██    ██    ██ ██  ██ ██ ██
  ████   ███████ ███████ ██      ██          ██     ██████  ██   ████ ███████

EOF
    printf "\033[0m"
    printf "  \033[2mUnified Triton kernel tuning for vLLM on NVIDIA GPUs\033[0m\n\n"
}

# ── Argument parsing ────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)           TP="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --dtype)        DTYPE="$2"; shift 2 ;;
        -t|--target)    CONTAINER="$2"; shift 2 ;;
        --deploy)       DO_DEPLOY=true; shift ;;
        --deploy-only)  DEPLOY_ONLY=true; shift ;;
        --attach)       ATTACH_ONLY=true; shift ;;
        --foreground|--no-tmux) USE_TMUX=false; shift ;;
        --tmux)         USE_TMUX=true; shift ;;  # back-compat (now default)
        --sync-mod)     SYNC_MOD=true; shift ;;
        --mod-dir)      MOD_DIR_OVERRIDE="$2"; shift 2 ;;
        --setup)        FORCE_SETUP=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --batch-size)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                BATCH_SIZES+=("$1"); shift
            done ;;
        --shapes)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SHAPES+=("$1"); shift
            done ;;
        -h|--help) show_help ;;
        --version) echo "vllm-tune $VERSION"; exit 0 ;;
        -*) die "Unknown flag: $1" ;;
        *)
            if [[ -z "$MODEL" ]]; then MODEL="$1"; shift
            else die "Unexpected argument: $1"; fi ;;
    esac
done

# --attach can work without a MODEL (finds any vllm-tune session)
if $ATTACH_ONLY; then
    if [[ -n "$MODEL" ]]; then
        SLUG=$(model_slug "$MODEL")
        SESSION="vllm-tune_${SLUG:0:30}"
    else
        # Find any vllm-tune tmux session
        SESSION=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^vllm-tune_' | head -1 || true)
        [[ -n "$SESSION" ]] || die "No active vllm-tune tmux sessions found."
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        exec tmux attach -t "$SESSION"
    else
        die "Session '$SESSION' not found. No tuning in progress."
    fi
fi

[[ -n "$MODEL" ]] || die "MODEL_ID is required. Run with --help for usage."
[[ "$MODE" =~ ^(moe|fp8|all)$ ]] || die "Invalid --mode '$MODE'. Use: moe, fp8, or all"

SLUG=$(model_slug "$MODEL")
MODEL_DIR="$CONFIG_HOME/$SLUG/tp${TP}"
CONFIGS_MOE="$MODEL_DIR/moe"
CONFIGS_FP8="$MODEL_DIR/fp8"
REPORT_DIR="$CONFIG_HOME/reports"

# ── tmux re-exec ────────────────────────────────────────────────────

# Skip tmux for quick operations or when already inside a tmux re-exec
if $DEPLOY_ONLY || $DRY_RUN || [[ "${_VLLM_TUNE_INSIDE_TMUX:-}" == "1" ]]; then
    USE_TMUX=false
fi

if $USE_TMUX; then
    command -v tmux &>/dev/null || die "tmux is required for background tuning. Install it or use --foreground."

    SESSION="vllm-tune_${SLUG:0:30}"
    # Rebuild the command with --foreground (runs inside tmux, no re-nesting)
    REEXEC_ARGS=("$MODEL" --tp "$TP" --mode "$MODE" --dtype "$DTYPE" -t "$CONTAINER" --foreground)
    [[ ${#BATCH_SIZES[@]} -gt 0 ]] && REEXEC_ARGS+=(--batch-size "${BATCH_SIZES[@]}")
    [[ ${#SHAPES[@]} -gt 0 ]] && REEXEC_ARGS+=(--shapes "${SHAPES[@]}")
    $DO_DEPLOY && REEXEC_ARGS+=(--deploy)
    if $SYNC_MOD; then
        REEXEC_ARGS+=(--sync-mod)
        [[ -n "$MOD_DIR_OVERRIDE" ]] && REEXEC_ARGS+=(--mod-dir "$MOD_DIR_OVERRIDE")
    fi

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        show_banner
        echo "  Tuning session '$SESSION' is already running."
        echo ""
        echo "  Attach:  vllm-tune.sh --attach"
        echo "     or:   tmux attach -t $SESSION"
        exit 0
    fi

    show_banner
    echo "  Starting tuning in tmux session: $SESSION"
    # Quote args properly for tmux
    QUOTED_ARGS=""
    for arg in "${REEXEC_ARGS[@]}"; do
        QUOTED_ARGS+=" $(printf '%q' "$arg")"
    done
    _VLLM_TUNE_INSIDE_TMUX=1 tmux new-session -d -s "$SESSION" \
        "$SCRIPT_DIR/vllm-tune.sh$QUOTED_ARGS; echo ''; echo 'Tuning complete. Press Enter to close.'; read"
    echo ""
    echo "  Attach:  vllm-tune.sh --attach"
    echo "     or:   tmux attach -t $SESSION"
    echo "  Detach:  Ctrl-b d"
    exit 0
fi

# ── Preflight checks ───────────────────────────────────────────────

if ! $DEPLOY_ONLY; then
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
        echo "" >&2
        echo "  Error: No running container named '$CONTAINER'." >&2
        echo "" >&2
        echo "  vLLM-Tune needs a running vLLM container to tune kernels inside." >&2
        echo "  Start one first, e.g.:" >&2
        echo "    launch-cluster.sh ...    # spark-vllm-docker" >&2
        echo "" >&2
        if [[ "$CONTAINER" == "vllm_node" ]]; then
            echo "  Using a different container name? Pass -t <name>" >&2
        fi
        exit 1
    fi
    command -v jq &>/dev/null || die "jq is required. Install: sudo apt install jq"
fi

mkdir -p "$CONFIGS_MOE" "$CONFIGS_FP8" "$REPORT_DIR"
detect_vllm_paths

# ── Banner ──────────────────────────────────────────────────────────

show_banner

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model:     $MODEL"
echo "  TP:        $TP"
echo "  Mode:      $MODE"
echo "  Container: $CONTAINER"
echo "  Configs:   $MODEL_DIR/"
if $DEPLOY_ONLY; then echo "  Action:    deploy-only"; fi
if $DRY_RUN;     then echo "  ⚠ DRY RUN — no changes will be made"; fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# ── Deploy function ─────────────────────────────────────────────────

deploy_configs() {
    local src_dir="$1" dst_path="$2" label="$3"
    local count=0

    if [[ ! -d "$src_dir" ]] || ! compgen -G "$src_dir/*.json" >/dev/null 2>&1; then
        warn "No $label configs to deploy from $src_dir"
        return
    fi

    echo "  Deploying $label configs → $CONTAINER:$dst_path"
    docker exec "$CONTAINER" mkdir -p "$dst_path" 2>/dev/null || true

    for cfg in "$src_dir"/*.json; do
        [[ -f "$cfg" ]] || continue
        local name
        name=$(basename "$cfg")
        if $DRY_RUN; then
            echo "    [dry-run] would copy: $name"
        else
            docker cp "$cfg" "$CONTAINER:$dst_path/$name"
            echo "    [ok] $name"
        fi
        count=$((count + 1))
    done
    ok "Deployed $count $label config(s)"
}

deploy_all() {
    [[ "$MODE" == "all" || "$MODE" == "moe" ]] && \
        deploy_configs "$CONFIGS_MOE" "$VLLM_MOE_PATH" "MoE"
    [[ "$MODE" == "all" || "$MODE" == "fp8" ]] && \
        deploy_configs "$CONFIGS_FP8" "$VLLM_FP8_PATH" "FP8"
}

# ── Deploy-only mode ────────────────────────────────────────────────

if $DEPLOY_ONLY; then
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
        echo "" >&2
        echo "  Error: No running container named '$CONTAINER'." >&2
        echo "" >&2
        echo "  Start a vLLM container first, then re-run with --deploy-only." >&2
        if [[ "$CONTAINER" == "vllm_node" ]]; then
            echo "  Using a different container name? Pass -t <name>" >&2
        fi
        echo "" >&2
        exit 1
    fi

    deploy_all

    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  \033[1mNext steps:\033[0m\n"
    echo "  Restart vLLM to pick up the new configs:"
    echo "    docker restart $CONTAINER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
fi

# ── Tuning ──────────────────────────────────────────────────────────

BS_ARGS=()
[[ ${#BATCH_SIZES[@]} -gt 0 ]] && BS_ARGS=(--batch-size "${BATCH_SIZES[@]}")

SHAPE_ARGS=()
[[ ${#SHAPES[@]} -gt 0 ]] && SHAPE_ARGS=(--shapes "${SHAPES[@]}")

TUNING_START=$SECONDS
MOE_OK=false
FP8_OK=false

# ── MoE tuning ──────────────────────────────────────────────────────

if [[ "$MODE" == "all" || "$MODE" == "moe" ]]; then
    echo "┌─────────────────────────────────────────────────────────────────"
    printf "│ \033[1mPhase 1: MoE Kernel Tuning\033[0m\n"
    echo "└─────────────────────────────────────────────────────────────────"

    MOE_SCRIPT="$TUNE_SCRIPTS_DIR/tune-moe.sh"
    [[ -x "$MOE_SCRIPT" ]] || die "MoE tuning script not found: $MOE_SCRIPT"

    if $DRY_RUN; then
        echo "  [dry-run] $MOE_SCRIPT $MODEL --tp $TP --dtype $DTYPE ${BS_ARGS[*]+${BS_ARGS[*]}}"
    else
        CONFIGS_DIR="$CONFIGS_MOE" \
        HOST_BACKUP_DIR="$CONFIG_HOME/backups/$SLUG/moe" \
        CONTAINER="$CONTAINER" \
        "$MOE_SCRIPT" "$MODEL" --tp "$TP" --dtype "$DTYPE" ${BS_ARGS[@]+"${BS_ARGS[@]}"} && MOE_OK=true || true
    fi
    echo
fi

# ── FP8 tuning ──────────────────────────────────────────────────────

if [[ "$MODE" == "all" || "$MODE" == "fp8" ]]; then
    echo "┌─────────────────────────────────────────────────────────────────"
    printf "│ \033[1mPhase 2: FP8 Dense GEMM Tuning\033[0m\n"
    echo "└─────────────────────────────────────────────────────────────────"

    FP8_SCRIPT="$TUNE_SCRIPTS_DIR/tune-fp8.sh"
    [[ -x "$FP8_SCRIPT" ]] || die "FP8 tuning script not found: $FP8_SCRIPT"

    if $DRY_RUN; then
        echo "  [dry-run] $FP8_SCRIPT $MODEL --tp $TP ${SHAPE_ARGS[*]+${SHAPE_ARGS[*]}} ${BS_ARGS[*]+${BS_ARGS[*]}}"
    else
        CONFIGS_DIR="$CONFIGS_FP8" \
        HOST_BACKUP_DIR="$CONFIG_HOME/backups/$SLUG/fp8" \
        CONTAINER="$CONTAINER" \
        "$FP8_SCRIPT" "$MODEL" --tp "$TP" ${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"} ${BS_ARGS[@]+"${BS_ARGS[@]}"} && FP8_OK=true || true
    fi
    echo
fi

TUNING_ELAPSED=$(( SECONDS - TUNING_START ))

# ── Deploy after tuning ─────────────────────────────────────────────

if $DO_DEPLOY && ! $DRY_RUN; then
    echo "┌─────────────────────────────────────────────────────────────────"
    printf "│ \033[1mDeploying configs to container\033[0m\n"
    echo "└─────────────────────────────────────────────────────────────────"

    deploy_all
    echo
fi

# ── Sync to mod directories ─────────────────────────────────────────

# Discover candidate mod locations on this system.
# Checks config.json spark_vllm_docker first, then hardcoded fallbacks.
_discover_mod_candidates() {
    local candidates=()
    # Check config.json for spark_vllm_docker path
    local cfg_spark
    cfg_spark=$(cfg_get spark_vllm_docker)
    if [[ -n "$cfg_spark" && -d "$cfg_spark/mods" ]]; then
        candidates+=("$cfg_spark/mods/vllm-tune")
    fi
    for candidate in \
        "$HOME/scripts/vllm/recipes/mods/vllm-tune" \
        "$HOME/spark-vllm-docker/mods/vllm-tune" \
        "$HOME/code/spark-vllm-docker/mods/vllm-tune"; do
        # Parent mods/ dir must exist, skip duplicates
        if [[ -d "$(dirname "$candidate")" ]]; then
            local dup=false
            for c in "${candidates[@]+${candidates[@]}}"; do
                [[ "$c" == "$candidate" ]] && { dup=true; break; }
            done
            $dup || candidates+=("$candidate")
        fi
    done
    echo "${candidates[@]}"
}

# Interactive first-time setup: let user choose sync targets.
_run_sync_setup() {
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  vLLM-Tune: first-time --sync-mod setup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "  Where should --sync-mod install configs?"
    echo "  Select the mod directories used by your launcher(s)."
    echo

    local candidates
    read -ra candidates <<< "$(_discover_mod_candidates)"
    local selected=()

    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "  No known mod directories found."
        echo -n "  Enter a custom path: "
        read -r custom_path
        [[ -n "$custom_path" ]] && selected+=("$custom_path")
    else
        for i in "${!candidates[@]}"; do
            local label="${candidates[$i]}"
            # Add helpful context
            if [[ "$label" == *spark-vllm-docker* ]]; then
                label="$label  (spark-vllm-docker)"
            elif [[ "$label" == *scripts/vllm* ]]; then
                label="$label  (start-vllm.sh)"
            fi
            echo -n "  [$((i+1))] $label — sync here? [Y/n] "
            read -r answer
            if [[ -z "$answer" || "$answer" =~ ^[Yy] ]]; then
                selected+=("${candidates[$i]}")
            fi
        done
    fi

    if [[ ${#selected[@]} -eq 0 ]]; then
        warn "No targets selected. --sync-mod will have nowhere to write."
        return 1
    fi

    # Save selection to config.json
    cfg_set_array "sync_targets" "${selected[@]}"
    echo
    ok "Saved sync targets to config.json"
    echo "  Run with --setup to change this later."
    echo
}

# Load MOD_DIRS from config.json, explicit override, or run setup.
_resolve_mod_dirs() {
    MOD_DIRS=()

    # Explicit --mod-dir always wins (single target, no setup)
    if [[ -n "${MOD_DIR_OVERRIDE:-}" ]]; then
        MOD_DIRS=("$MOD_DIR_OVERRIDE")
        return
    fi

    # --setup: force interactive chooser
    if ${FORCE_SETUP:-false}; then
        _run_sync_setup || return
    fi

    # Load saved targets from config.json
    mapfile -t MOD_DIRS < <(cfg_get_array "sync_targets")
    [[ ${#MOD_DIRS[@]} -gt 0 ]] && return

    # No saved config → run first-time setup
    _run_sync_setup || return

    # Reload after setup
    mapfile -t MOD_DIRS < <(cfg_get_array "sync_targets")
}

# Sync configs to a single mod directory.
# Bootstraps from bundled mod/run.sh if the directory doesn't exist.
_sync_one_mod_dir() {
    local mod_dir="$1"

    # Bootstrap: create mod directory from bundled mod/ if it doesn't exist
    if [[ ! -d "$mod_dir" ]]; then
        if [[ -f "$SCRIPT_DIR/mod/run.sh" ]]; then
            info "Creating $mod_dir from bundled template..."
            mkdir -p "$mod_dir"
            install -m 0755 "$SCRIPT_DIR/mod/run.sh" "$mod_dir/run.sh"
        else
            warn "Cannot create $mod_dir — no bundled mod/run.sh"
            return
        fi
    fi

    # Keep run.sh in sync with the repo version
    if [[ -f "$SCRIPT_DIR/mod/run.sh" ]]; then
        install -m 0755 "$SCRIPT_DIR/mod/run.sh" "$mod_dir/run.sh"
    fi

    # Sync JSON configs
    mkdir -p "$mod_dir/configs" "$mod_dir/fp8-configs"
    rsync -a --include='*.json' --exclude='*' "$CONFIGS_MOE/" "$mod_dir/configs/" 2>/dev/null || true
    rsync -a --include='*.json' --exclude='*' "$CONFIGS_FP8/" "$mod_dir/fp8-configs/" 2>/dev/null || true
    ok "Synced to $mod_dir"
}

if $SYNC_MOD && ! $DRY_RUN; then
    _resolve_mod_dirs
    for _mod_dir in "${MOD_DIRS[@]+${MOD_DIRS[@]}}"; do
        info "Syncing to: $_mod_dir"
        _sync_one_mod_dir "$_mod_dir"
    done
fi

# ── Write metadata ──────────────────────────────────────────────────

if ! $DRY_RUN; then
    META_FILE="$MODEL_DIR/metadata.json"
    VLLM_VERSION=$(docker exec "$CONTAINER" python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    cat > "$META_FILE" <<EOF
{
    "model": "$MODEL",
    "tp": $TP,
    "dtype": "$DTYPE",
    "tuned_at": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "container": "$CONTAINER",
    "mode": "$MODE",
    "vllm_version": "$VLLM_VERSION",
    "moe_configs": $(find "$CONFIGS_MOE" -name '*.json' 2>/dev/null | wc -l),
    "fp8_configs": $(find "$CONFIGS_FP8" -name '*.json' 2>/dev/null | wc -l)
}
EOF
fi

# ── Summary report ──────────────────────────────────────────────────

moe_count=$(find "$CONFIGS_MOE" -name '*.json' 2>/dev/null | wc -l)
fp8_count=$(find "$CONFIGS_FP8" -name '*.json' 2>/dev/null | wc -l)

REPORT="$REPORT_DIR/tune-$(date +%Y%m%d-%H%M%S)-${SLUG}-tp${TP}.md"
if ! $DRY_RUN; then
    cat > "$REPORT" <<EOF
# vLLM-Tune Report

Generated: $(date -Iseconds)

| Parameter | Value |
|-----------|-------|
| Model | \`$MODEL\` |
| TP | $TP |
| Mode | $MODE |
| Container | $CONTAINER |
| Duration | $((TUNING_ELAPSED/60))m $((TUNING_ELAPSED%60))s |

## Config Files

- **MoE:** $moe_count config(s) in \`$CONFIGS_MOE\`
- **FP8:** $fp8_count config(s) in \`$CONFIGS_FP8\`

## Deploy Command

\`\`\`bash
vllm-tune.sh $MODEL --tp $TP --deploy-only -t $CONTAINER
\`\`\`
EOF
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  \033[1mDone!\033[0m  %dm %ds total\n" "$((TUNING_ELAPSED/60))" "$((TUNING_ELAPSED%60))"
echo "  MoE configs: $moe_count  |  FP8 configs: $fp8_count"
echo "  Report: $REPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
printf "  \033[1mNext steps:\033[0m\n"
if ! $DO_DEPLOY; then
    echo "  1. Deploy to running container (immediate, lost on stopcluster):"
    echo "       vllm-tune.sh $MODEL --tp $TP --deploy-only -t $CONTAINER"
fi
echo "  2. Sync to mod (persistent — survives stop/start):"
echo "       vllm-tune.sh $MODEL --tp $TP --sync-mod"
echo "     Then re-enable vllm-tune in your recipe's VLLM_MODS."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
