#!/bin/bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────
# tune-moe.sh — Tune MoE fused expert dispatch kernel configs
# ─────────────────────────────────────────────────────────────────────
#
# Benchmarks Triton fused_moe kernels across batch sizes to find optimal
# BLOCK_SIZE_M/N/K and GROUP_SIZE_M parameters for each (E, N) shape.
#
# Usage:
#   ./tune-moe.sh <MODEL_ID> [--tp <TP_SIZE>] [--batch-size <SIZE>...] [--dtype <DTYPE>]
#
# Examples:
#   ./tune-moe.sh Qwen/Qwen3.6-35B-A3B-FP8                         # tune all 18 default sizes
#   ./tune-moe.sh Qwen/Qwen3.6-35B-A3B-FP8 --batch-size 512 1024   # tune only specific sizes
#   ./tune-moe.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 1                  # single GPU
#
# Process:
#   1. Tunes one batch size at a time inside the running container
#   2. Retries once on failure before moving on
#   3. Copies results out to the host after each batch size (crash-safe)
#   4. Merges new entries into existing configs via jq
#   5. Writes a tuning-report.md alongside the configs
#
# Environment overrides (set by vllm-tune.sh when orchestrating):
#   CONFIGS_DIR     — Final config output directory
#   HOST_BACKUP_DIR — Host-side incremental backup directory
#   CONTAINER       — Docker container name
#
# Requirements:
#   - A running vllm_node container with GPU access
#   - jq installed on the host
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# ── Configuration ───────────────────────────────────────────────────

CONFIGS_DIR="${CONFIGS_DIR:-$SCRIPT_DIR/configs}"
CONTAINER="${CONTAINER:-vllm_node}"
CONTAINER_SAVE_DIR="/tmp/moe-configs-tuning"
HOST_BACKUP_DIR="${HOST_BACKUP_DIR:-/tmp/moe-configs-backup}"
MAX_RETRIES=1

# Defaults
TP=2
DTYPE="fp8_w8a8"
MODEL=""
BATCH_SIZES=()

# ── Argument parsing ────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)         TP="$2"; shift 2 ;;
        --dtype)      DTYPE="$2"; shift 2 ;;
        --batch-size)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                BATCH_SIZES+=("$1"); shift
            done ;;
        -h|--help)
            sed -n '2,/^$/{ s/^# \?//; p }' "$0"
            exit 0 ;;
        -*)  echo "Unknown flag: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; exit 1
            fi ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: MODEL_ID is required" >&2
    echo "Usage: $(basename "$0") <MODEL_ID> [--tp <N>] [--batch-size <SIZE>...]" >&2
    exit 1
fi

# Default batch sizes (matches vLLM's benchmark_moe.py defaults)
if [[ ${#BATCH_SIZES[@]} -eq 0 ]]; then
    BATCH_SIZES=(1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096)
fi

# ── Pre-flight checks ──────────────────────────────────────────────

preflight

# ── Tuning function ─────────────────────────────────────────────────

# Run benchmark_moe.py for a single batch size inside the container.
# Uses tini as subreaper (if available) to prevent zombie accumulation.
run_tune() {
    local bs=$1
    docker exec "$CONTAINER" $INIT_WRAPPER bash -c \
        "rm -rf /tmp/vllm-bench && \
         git clone --depth 1 --filter=blob:none --sparse \
           https://github.com/vllm-project/vllm.git /tmp/vllm-bench 2>/dev/null && \
         cd /tmp/vllm-bench && git sparse-checkout set benchmarks 2>/dev/null && \
         sed -i 's/\"DeepseekV3ForCausalLM\",/\"DeepseekV3ForCausalLM\", \"DeepseekV4ForCausalLM\",/g' benchmarks/kernels/benchmark_moe.py && \
         python3 benchmarks/kernels/benchmark_moe.py \
           --model $MODEL \
           --tp-size $TP \
           --dtype $DTYPE \
           --tune \
           --batch-size $bs \
           --save-dir $CONTAINER_SAVE_DIR/"
}

# ── Main loop ───────────────────────────────────────────────────────

TOTAL=${#BATCH_SIZES[@]}
COMPLETED=0
FAILED=()
SUCCEEDED=()
SKIPPED=()
declare -A TIMINGS

# Check if a batch size is already tuned in existing config files.
# Looks for the batch size key inside any matching config JSON.
# Returns 0 (already tuned) or 1 (needs tuning).
_batch_already_tuned() {
    local bs="$1"
    for cfg in "$CONFIGS_DIR"/*.json; do
        [[ -f "$cfg" ]] || continue
        if jq -e --arg k "$bs" 'has($k)' "$cfg" >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  \033[1mMoE Kernel Tuning\033[0m\n"
echo "  Model:       $MODEL"
echo "  TP size:     $TP"
echo "  Dtype:       $DTYPE"
echo "  Batch sizes: ${BATCH_SIZES[*]}"
echo "  Retries:     $MAX_RETRIES per batch size"
echo "  Backup dir:  $HOST_BACKUP_DIR"
echo "  Configs dir: $CONFIGS_DIR"
if [[ -n "$INIT_WRAPPER" ]]; then
    echo "  Subreaper:   ${INIT_WRAPPER%% *}"
else
    echo "  Subreaper:   ⚠ none (zombies may accumulate — consider --init)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

TUNING_START=$SECONDS

for BS in "${BATCH_SIZES[@]}"; do
    COMPLETED=$((COMPLETED + 1))

    # Skip batch sizes that are already tuned
    if _batch_already_tuned "$BS"; then
        printf "  ⏭ [%d/%d] batch_size=%s — already tuned, skipping\n" "$COMPLETED" "$TOTAL" "$BS"
        SKIPPED+=("$BS")
        continue
    fi

    echo "┌─────────────────────────────────────────────────────────────────"
    printf "│ \033[1m[%d/%d] Tuning batch_size=%s\033[0m\n" "$COMPLETED" "$TOTAL" "$BS"
    echo "└─────────────────────────────────────────────────────────────────"

    START_TIME=$SECONDS

    if run_with_retry "batch_size=$BS" run_tune "$BS"; then
        ELAPSED=$(( SECONDS - START_TIME ))
        TIMINGS[$BS]=$ELAPSED
        printf "  ✅ batch_size=%s completed in %s\n" "$BS" "$(fmt_time $ELAPSED)"
        SUCCEEDED+=("$BS")
        merge_results "moe-configs-tuning"
    else
        ELAPSED=$(( SECONDS - START_TIME ))
        TIMINGS[$BS]=$ELAPSED
        printf "  ❌ batch_size=%s \033[31mfailed after %d attempts\033[0m (%s)\n" \
            "$BS" "$((MAX_RETRIES + 1))" "$(fmt_time $ELAPSED)"
        FAILED+=("$BS")
    fi

    post_round
done

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "  ⏭ Skipped ${#SKIPPED[@]} already-tuned batch size(s): ${SKIPPED[*]}"
    echo
fi

TUNING_ELAPSED=$(( SECONDS - TUNING_START ))

# ── Report & summary ───────────────────────────────────────────────

rerun_cmd=""
[[ ${#FAILED[@]} -gt 0 ]] && rerun_cmd="./tune-moe.sh $MODEL --tp $TP --dtype $DTYPE --batch-size ${FAILED[*]}"

generate_report \
    "MoE Tuning Report" \
    "Batch Size" \
    "$TUNING_ELAPSED" \
    "${BATCH_SIZES[*]}" \
    "${FAILED[*]+${FAILED[*]}}" \
    "$rerun_cmd"

cleanup_zombies

print_summary \
    "${#SUCCEEDED[@]}" "$TOTAL" "$TUNING_ELAPSED" "batch sizes" \
    "${FAILED[*]+${FAILED[*]}}" "$rerun_cmd"
