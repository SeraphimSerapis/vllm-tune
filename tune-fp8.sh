#!/bin/bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────
# tune-fp8.sh — Tune W8A8 Block FP8 dense GEMM kernel configs
# ─────────────────────────────────────────────────────────────────────
#
# Counterpart to tune-moe.sh — optimizes the FP8 block-scaled matmul
# kernels used by attention projections, shared experts, and other dense
# linear layers. Without these configs, vLLM prints:
#   "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!"
#
# These configs help ANY model using FP8 quantization (both MoE and dense).
#
# Usage:
#   ./tune-fp8.sh <MODEL_ID> [--tp <TP_SIZE>] [--batch-size <SIZE>...]
#   ./tune-fp8.sh --shapes 6144,2048 2048,2048 [--batch-size <SIZE>...]
#
# Examples:
#   ./tune-fp8.sh Qwen/Qwen3.6-35B-A3B-FP8                         # auto-detect shapes
#   ./tune-fp8.sh deepseek-ai/DeepSeek-V3 --tp 2                    # works with any FP8 model
#   ./tune-fp8.sh Qwen/Qwen3.6-35B-A3B-FP8 --batch-size 1 64 256   # specific batch sizes
#   ./tune-fp8.sh --shapes 6144,2048 2048,2048                      # explicit shapes
#
# Process:
#   1. Detects required (N,K) shapes from model config (or accepts --shapes)
#   2. Tunes one shape at a time inside the container (crash-safe)
#   3. Retries once on failure before moving on
#   4. Copies results out to the host after each shape
#   5. Merges new entries into existing configs via jq
#   6. Writes a tuning-report.md alongside the configs
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
CONTAINER_SAVE_DIR="/tmp/fp8-configs-tuning"
HOST_BACKUP_DIR="${HOST_BACKUP_DIR:-/tmp/fp8-configs-backup}"
MAX_RETRIES=1

# Defaults
TP=2
MODEL=""
BATCH_SIZES=()
SHAPES=()

# ── Argument parsing ────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)         TP="$2"; shift 2 ;;
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

if [[ -z "$MODEL" && ${#SHAPES[@]} -eq 0 ]]; then
    echo "Error: MODEL_ID or --shapes is required" >&2
    echo "Usage: $(basename "$0") <MODEL_ID> [--tp <N>] [--batch-size <SIZE>...]" >&2
    echo "   or: $(basename "$0") --shapes N,K [N,K ...] [--batch-size <SIZE>...]" >&2
    exit 1
fi

# Default batch sizes
if [[ ${#BATCH_SIZES[@]} -eq 0 ]]; then
    BATCH_SIZES=(1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096)
fi

# ── Pre-flight checks ──────────────────────────────────────────────

preflight

# ── Shape detection ─────────────────────────────────────────────────

# Auto-detect (N,K) shapes from the model's architecture config.
if [[ ${#SHAPES[@]} -eq 0 && -n "$MODEL" ]]; then
    echo "  🔍 Auto-detecting weight shapes for $MODEL (TP=$TP)..."
    DETECTED=$(docker exec "$CONTAINER" python3 -c "$(cat "$SCRIPT_DIR/lib/detect.py")" "$MODEL" --tp "$TP" --mode shapes 2>&1) || true
    
    # Parse JSON output from detect.py (filter out vLLM/HF noise)
    JSON_LINE=$(echo "$DETECTED" | grep '^{' || true)
    if [[ -z "$JSON_LINE" ]]; then
        echo "  ⚠ Detection failed: $DETECTED" >&2
    else
        mapfile -t SHAPES < <(echo "$JSON_LINE" | jq -r '.shapes[]')
    fi

    if [[ ${#SHAPES[@]} -eq 0 ]]; then
        echo "  ❌ Could not detect shapes. Use --shapes N,K to specify manually." >&2
        exit 1
    fi
    echo "  ✅ Detected ${#SHAPES[@]} shapes: ${SHAPES[*]}"
fi

# ── Tuning function ─────────────────────────────────────────────────

# Run the FP8 block-scaled GEMM benchmark for a single (N,K) shape
# across all batch sizes. Uses Triton autotuning over a grid of
# BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, and num_stages.
run_tune_shape() {
    local n=$1 k=$2
    local bs_python
    bs_python=$(IFS=,; echo "${BATCH_SIZES[*]}")

    docker exec "$CONTAINER" $INIT_WRAPPER python3 -c "
import json, os, sys, time
import torch
from tqdm import tqdm
from vllm.model_executor.layers.quantization.utils.fp8_utils import _w8a8_triton_block_scaled_mm
from vllm.platforms import current_platform
from vllm.triton_utils import triton

N, K = $n, $k
block_n, block_k = 128, 128
batch_sizes = [$bs_python]
save_dir = '$CONTAINER_SAVE_DIR'
os.makedirs(save_dir, exist_ok=True)

device_name = current_platform.get_device_name().replace(' ', '_')
fp8_info = torch.finfo(torch.float8_e4m3fn)
fp8_max, fp8_min = fp8_info.max, fp8_info.min

# Build search space (BLOCK_SIZE_M/N/K × GROUP_SIZE_M × num_warps × num_stages)
configs = []
for ns in [2, 3, 4, 5]:
    for bm in [16, 32, 64, 128, 256]:
        for bk in [64, 128]:
            for bn in [32, 64, 128, 256]:
                for nw in [4, 8]:
                    for gm in [1, 16, 32, 64]:
                        if block_k % bk == 0:
                            configs.append({
                                'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn,
                                'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': gm,
                                'num_warps': nw, 'num_stages': ns,
                            })

def bench(A, B, As, Bs, config, num_iters=10):
    M = A.shape[0]
    C = A.new_empty((M, N), dtype=torch.float16)
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    def run():
        _w8a8_triton_block_scaled_mm[grid](
            A, B, C, As, Bs, M, N, K, block_n, block_k,
            A.stride(0), A.stride(1), B.stride(1), B.stride(0),
            C.stride(0), C.stride(1), As.stride(0), As.stride(1),
            Bs.stride(1), Bs.stride(0), **config)
    # warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / num_iters

results = {}
n_tiles = (N + block_n - 1) // block_n
k_tiles = (K + block_k - 1) // block_k

for M in batch_sizes:
    A = ((torch.rand(M, K, device='cuda') - 0.5) * 2 * fp8_max).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    B = ((torch.rand(N, K, device='cuda') - 0.5) * 2 * fp8_max).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    As = torch.rand(M, k_tiles, device='cuda', dtype=torch.float32) * 1e-2
    Bs = torch.rand(n_tiles, k_tiles, device='cuda', dtype=torch.float32) * 1e-2

    best_time, best_cfg = float('inf'), None
    for cfg in tqdm(configs, desc=f'M={M}, N={N}, K={K}'):
        try:
            t = bench(A, B, As, Bs, cfg)
        except triton.runtime.autotuner.OutOfResources:
            continue
        except Exception:
            continue
        if t < best_time:
            best_time, best_cfg = t, cfg

    if best_cfg:
        results[str(M)] = best_cfg
        print(f'  batch_size={M}: {best_time:.2f}ms — {best_cfg}')
    else:
        print(f'  batch_size={M}: FAILED (no valid config)')
    # Free GPU memory between batch sizes
    del A, B, As, Bs
    torch.cuda.empty_cache()

filename = f'N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json'
filepath = os.path.join(save_dir, filename)
with open(filepath, 'w') as f:
    json.dump(results, f, indent=4)
    f.write('\n')
print(f'Saved: {filepath}')
"
}

# ── Main loop ───────────────────────────────────────────────────────

TOTAL=${#SHAPES[@]}
COMPLETED=0
FAILED=()
SUCCEEDED=()
SKIPPED=()
declare -A TIMINGS

# Check if a shape is already fully tuned for all requested batch sizes.
# Reads the device name from any existing config file, then checks if
# the specific shape's config file exists and contains all batch sizes.
# Returns 0 (fully tuned) or 1 (needs tuning).
_shape_already_tuned() {
    local n="$1" k="$2"
    # Find matching config file (we don't know device_name yet, so glob it)
    local pattern="$CONFIGS_DIR/N=${n},K=${k},device_name=*,dtype=fp8_w8a8,block_shape=*.json"
    local cfg
    for cfg in $pattern; do
        [[ -f "$cfg" ]] || continue
        # Check if all requested batch sizes are present as keys
        local all_present=true
        for bs in "${BATCH_SIZES[@]}"; do
            if ! jq -e --arg k "$bs" 'has($k)' "$cfg" >/dev/null 2>&1; then
                all_present=false
                break
            fi
        done
        if $all_present; then
            return 0
        fi
    done
    return 1
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  \033[1mW8A8 Block FP8 Kernel Tuning\033[0m\n"
[[ -n "$MODEL" ]] && echo "  Model:       $MODEL"
echo "  TP size:     $TP"
echo "  Shapes:      ${SHAPES[*]}"
echo "  Batch sizes: ${BATCH_SIZES[*]}"
echo "  Retries:     $MAX_RETRIES per shape"
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

for SHAPE in "${SHAPES[@]}"; do
    N="${SHAPE%%,*}"
    K="${SHAPE##*,}"
    COMPLETED=$((COMPLETED + 1))

    # Skip shapes that are already fully tuned
    if _shape_already_tuned "$N" "$K"; then
        printf "  ⏭ [%d/%d] N=%s,K=%s — already tuned, skipping\n" "$COMPLETED" "$TOTAL" "$N" "$K"
        SKIPPED+=("${N},${K}")
        continue
    fi

    echo "┌─────────────────────────────────────────────────────────────────"
    printf "│ \033[1m[%d/%d] Tuning N=%s, K=%s\033[0m\n" "$COMPLETED" "$TOTAL" "$N" "$K"
    echo "└─────────────────────────────────────────────────────────────────"

    START_TIME=$SECONDS

    if run_with_retry "N=$N,K=$K" run_tune_shape "$N" "$K"; then
        ELAPSED=$(( SECONDS - START_TIME ))
        TIMINGS["${N},${K}"]=$ELAPSED
        printf "  ✅ N=%s,K=%s completed in %s\n" "$N" "$K" "$(fmt_time $ELAPSED)"
        SUCCEEDED+=("${N},${K}")
        merge_results "fp8-configs-tuning"
    else
        ELAPSED=$(( SECONDS - START_TIME ))
        TIMINGS["${N},${K}"]=$ELAPSED
        printf "  ❌ N=%s,K=%s \033[31mfailed after %d attempts\033[0m (%s)\n" \
            "$N" "$K" "$((MAX_RETRIES + 1))" "$(fmt_time $ELAPSED)"
        FAILED+=("${N},${K}")
    fi

    post_round
done

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "  ⏭ Skipped ${#SKIPPED[@]} already-tuned shape(s): ${SKIPPED[*]}"
    echo
fi

TUNING_ELAPSED=$(( SECONDS - TUNING_START ))

# ── Report & summary ───────────────────────────────────────────────

rerun_cmd=""
[[ ${#FAILED[@]} -gt 0 ]] && rerun_cmd="./tune-fp8.sh --shapes ${FAILED[*]} --batch-size ${BATCH_SIZES[*]}"

generate_report \
    "FP8 Dense GEMM Tuning Report" \
    "Shape (N,K)" \
    "$TUNING_ELAPSED" \
    "${SHAPES[*]}" \
    "${FAILED[*]+${FAILED[*]}}" \
    "$rerun_cmd"

cleanup_zombies

print_summary \
    "${#SUCCEEDED[@]}" "$TOTAL" "$TUNING_ELAPSED" "shapes" \
    "${FAILED[*]+${FAILED[*]}}" "$rerun_cmd"
