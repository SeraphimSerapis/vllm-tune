# vLLM-Tune

```
██    ██ ██      ██      ███    ███       ████████ ██    ██ ███    ██ ███████
██    ██ ██      ██      ████  ████          ██    ██    ██ ████   ██ ██
██    ██ ██      ██      ██ ████ ██ ███████  ██    ██    ██ ██ ██  ██ █████
 ██  ██  ██      ██      ██  ██  ██          ██    ██    ██ ██  ██ ██ ██
  ████   ███████ ███████ ██      ██          ██     ██████  ██   ████ ███████
```

Unified CLI for tuning vLLM Triton kernel configs on NVIDIA GPUs.
Developed and tested on DGX Spark (GB10).

Consolidates MoE and FP8 dense GEMM kernel tuning into a single command,
with tmux support for long-running jobs and `docker cp` deployment.

## Why?

Without tuned kernel configs, vLLM falls back to default heuristics and prints warnings like:

```
WARNING Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!
WARNING Using default MoE config. Performance might be sub-optimal!
```

### Performance Impact

Benchmarked on **Qwen/Qwen3.6-35B-A3B-FP8** (TP=2, NVIDIA GB10 × 2, vLLM v0.19.2) with `llama-benchy`:

| Test | Default (t/s) | Tuned (t/s) | Δ | Improvement |
|------|---------------|-------------|---|-------------|
| **pp2048** (prefill) | 4,677 ± 2,150 | **7,406 ± 19** | +2,729 | **+58%** throughput, **50% lower** TTFT |
| **tg128** (decode) | 74.4 ± 0.3 | **81.5 ± 0.0** | +7.1 | **+9.5%** tokens/sec |
| **pp2048 @ d4096** | 7,840 ± 138 | **7,896 ± 9** | +56 | **15× lower** variance |
| **tg128 @ d4096** | 73.5 ± 0.3 | **80.5 ± 0.1** | +7.0 | **+9.5%** tokens/sec |
| **pp2048 @ d8192** | 8,186 ± 28 | **8,209 ± 3** | +23 | **9× lower** variance |
| **tg128 @ d8192** | 72.7 ± 0.4 | **79.9 ± 0.1** | +7.2 | **+9.9%** tokens/sec |

**Key takeaways:**
- **Prefill throughput** jumps **+58%** (4.7k → 7.4k t/s) with dramatically lower variance
- **Decode speed** gains **~10%** consistently across all context depths
- **Stability** improves massively — default pp2048 has ±2,150 variance vs ±19 when tuned
- At longer contexts (d8192), decode drops from 82→80 t/s — but tuned stays **7 t/s faster**

In practical terms: a 2048-token prefill completes in **278ms tuned** vs **561ms default** — that's **283ms saved on every first response**.

## Installation

```bash
# One-liner install (interactive — detects spark-vllm-docker, sets up PATH):
curl -fsSL https://raw.githubusercontent.com/SeraphimSerapis/vllm-tune/main/install.sh | bash

# Or clone manually:
git clone https://github.com/SeraphimSerapis/vllm-tune.git ~/vllm-tune
```

## Prerequisites

vLLM-Tune runs tuning benchmarks **inside a running vLLM container**. Before running `vllm-tune.sh`, make sure you have:

1. A running vLLM Docker container (default name: `vllm_node`, override with `-t`)
   — e.g. launched via [spark-vllm-docker](https://github.com/NVIDIA/spark-vllm-docker)'s `launch-cluster.sh`
2. `jq` installed on the host
3. `tmux` (optional, for `--tmux` detachable sessions)
4. `sudo` access (optional, for cache clearing between tuning rounds)

## Quick Start

```bash
# Full tuning (MoE + FP8), deploy to container when done
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy

# MoE-only tuning in a tmux session (detach-safe for multi-hour runs)
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --mode moe --tmux

# FP8-only with custom shapes
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --mode fp8 --shapes 6144,2048 2048,2048

# Deploy existing configs to a running container
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy-only -t vllm_node
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--tp <N>` | Tensor parallelism size | `2` |
| `--mode <MODE>` | `moe`, `fp8`, or `all` (auto-skips MoE for dense models) | `all` |
| `--batch-size <S...>` | Custom batch sizes | 1–4096 (18 sizes) |
| `--shapes <N,K ...>` | Explicit FP8 shapes | auto-detected |
| `--dtype <DTYPE>` | MoE dtype | `fp8_w8a8` |
| `-t, --target <NAME>` | Container name | `vllm_node` |
| `--deploy` | Deploy configs after tuning | off |
| `--deploy-only` | Skip tuning, deploy existing | off |
| `--standalone` | Launch a dedicated tuning container (no inference needed) | off |
| `--image <IMAGE>` | Container image for `--standalone` | auto-detect |
| `--foreground` | Run in foreground instead of tmux | off |
| `--tmux` | Run in detachable tmux session | **on** (default) |
| `--sync-mod` | Sync to vllm-tune mod dir | off |
| `--export-sparkrun` | Copy configs to sparkrun's tuning cache | off |
| `--import-sparkrun` | Import configs from sparkrun's tuning cache | off |
| `--dry-run` | Show plan without executing | off |

## Config Store Layout

Configs are stored in the project directory, organized by model and TP:

```
configs/
└── qwen--qwen3.6-35b-a3b-fp8/
    ├── tp1/
    │   ├── metadata.json       # model, TP, dtype, timestamp, hostname
    │   ├── moe/                # E=...,N=... JSON files
    │   └── fp8/                # N=...,K=... JSON files
    └── tp2/
        ├── metadata.json
        ├── moe/
        └── fp8/
```

### What Gets Tuned?

| Tuning mode | What it tunes | Which models benefit? |
|-------------|--------------|----------------------|
| `--mode moe` | MoE fused expert dispatch kernels | MoE models only (e.g., Qwen-A3B, Mixtral) |
| `--mode fp8` | FP8 block-scaled dense GEMM kernels | **Any FP8 model** — MoE and dense alike |
| `--mode all` | Both | MoE FP8 models (gets maximum benefit) |

> **Dense FP8 models** (e.g., Llama-70B-FP8, Qwen3.6-27B-FP8) benefit from `--mode fp8` — it tunes
> the same Triton matmul kernels used in attention projections and FFN layers.
>
> **Auto-detection:** When using `--mode all` (the default), vLLM-Tune automatically
> detects whether the model has MoE layers. Dense models skip MoE tuning with an
> informative message and proceed directly to FP8 tuning — no errors, no wasted time.

### Scope: FP8 Models Only

This tool tunes Triton kernels used by vLLM's **FP8 (`fp8_w8a8`) code path** —
both MoE expert dispatch and dense GEMM matmuls. Configs are keyed by
architecture geometry (`E`, `N`, `K`, `dtype`, `device_name`), not model ID:

- **Same architecture + same TP** → identical configs, reusable across model variants
- **INT4 / AWQ / AutoRound** → use Marlin CUDA kernels, an entirely different backend.
  They don't generate or consume these configs at all.
- **Different TP values** → different effective shapes (N/TP), different configs

> **INT4/AWQ users:** Marlin kernels are already highly optimized CUDA code with no
> runtime config tuning. For INT4 performance on DGX Spark, the main levers are
> `--max-num-batched-tokens`, `--max-num-seqs`, attention backend selection, and
> ensuring `--load-format fastsafetensors` for faster startup.

### Pre-shipped Configs

The following tuned configs are included out of the box — no tuning required:

| Model | TP | MoE configs | FP8 configs | Tuned on |
|-------|-----|-------------|-------------|----------|
| `Qwen/Qwen3.6-35B-A3B-FP8` | 2 | 2 (E=256, N=256/512) | 5 (dense GEMM shapes) | NVIDIA GB10 |

To deploy them immediately:

```bash
# Deploy pre-shipped configs to running container
vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy-only

# Or sync to mod for persistent deployment
vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --sync-mod
```

## How It Works

1. **Tuning** — Delegates to `tune-moe.sh` and/or `tune-fp8.sh` (co-located in the project)
2. **Storage** — Saves tuned configs to `configs/<model>/tp<N>/` with metadata
3. **Deploy** — Uses `docker cp` to place configs in the vLLM container
4. **Track** — Writes `metadata.json` per model/TP combo for config provenance

## Deployment

There are two deployment paths. Both copy configs to the same vLLM paths:

- **MoE** → `.../vllm/model_executor/layers/fused_moe/configs/`
- **FP8** → `.../vllm/model_executor/layers/quantization/utils/configs/`

### Persistent: `--sync-mod` (recommended)

The `spark-vllm-docker` container runs with `--rm`, meaning **all container
files are destroyed on `stopcluster`**. To persist configs across stop/start
cycles, sync them to the `vllm-tune` mod:

```bash
vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --sync-mod
```

The mod is then applied automatically at container launch by `launch-cluster.sh`'s
`--apply-mod` mechanism. The mod's `run.sh` copies the JSON configs into the
correct vLLM paths inside the container.

**Important:** The mod directory location depends on which launch method you use:

| Launch method | Mod resolution | Default `--sync-mod` target |
|---------------|----------------|----------------------------|
| `start-vllm.sh` (custom wrapper) | `$RECIPES_DIR/mods/<name>` | `~/scripts/vllm/recipes/mods/vllm-tune` ✅ |
| `run-recipe.py` (standard) | `$SPARK_VLLM_DIR/<path>` | Needs `--mod-dir` override |
| `launch-cluster.sh --apply-mod` | Absolute path | Needs `--mod-dir` override |

For `start-vllm.sh` users (the default), `--sync-mod` works out of the box.
Then enable the mod in your recipe:

```bash
# In your recipe .sh file:
VLLM_MODS=(
    vllm-tune
    mods/drop-caches
)
```

For `run-recipe.py` users, point `--mod-dir` to the spark-vllm-docker mods path:

```bash
vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --sync-mod \
    --mod-dir ~/spark-vllm-docker/mods/vllm-tune

# In your recipe YAML:
mods:
  - mods/vllm-tune
```

### Immediate: `--deploy` / `--deploy-only`

For testing or hot-loading without a restart cycle, `docker cp` places configs
directly in the running container:

```bash
vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy-only
docker restart vllm_node   # vLLM must restart to load new configs
```

> **⚠️ These files are lost on `stopcluster`** since the container uses `--rm`.
> Always run `--sync-mod` to make configs permanent.

### Alternative: VLLM_TUNED_CONFIG_FOLDER

For MoE configs only, vLLM supports the `VLLM_TUNED_CONFIG_FOLDER` environment
variable. When set, vLLM prioritizes configs from that folder over its built-in
defaults. This can be useful for bind-mounting a host directory:

```bash
# In your recipe's VLLM_ENV:
VLLM_TUNED_CONFIG_FOLDER=/opt/vllm-tuned-configs
```

> **Note:** This only works for MoE fused_moe configs. FP8 dense GEMM configs
> (`fp8_utils.py`) do not check this env var — they must be placed in the
> hardcoded `configs/` directory via `docker cp` or image build.



## Environment Variables

| Variable | Description | Default |
|----------|-------------|--------|
| `VLLM_TUNE_HOME` | Override config store root | `<project>/configs` |
| `CONTAINER` | Docker container name | `vllm_node` |
| `CONFIGS_DIR` | Override config output directory | `<project>/configs` |
| `HOST_BACKUP_DIR` | Incremental backup directory | `/tmp/{moe,fp8}-configs-backup` |
| `MOD_DIR` | Override mod directory for `--sync-mod` | `~/scripts/vllm/recipes/mods/vllm-tune` |
| `PEER_NODES` | Space-separated SSH peers for cache clearing | _(none)_ |
| `DROP_CACHES_CMD` | Custom cache-drop command | `sync && echo 3 > /proc/sys/vm/drop_caches` |

## Using with spark-vllm-docker

vLLM-Tune integrates with [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker),
the community-standard Docker setup for vLLM on DGX Spark.

### Setup (symlink — keeps spark-vllm-docker repo clean)

```bash
# Symlink vllm-tune's mod into spark-vllm-docker:
ln -s ~/code/vllm-tune/mod ~/spark-vllm-docker/mods/vllm-tune

# Launch with the mod:
cd ~/spark-vllm-docker
./launch-cluster.sh --apply-mod mods/vllm-tune exec vllm serve \
    Qwen/Qwen3.6-35B-A3B-FP8 -tp 2 --load-format fastsafetensors ...
```

The symlink means `--sync-mod` updates flow through automatically and
`git pull` in spark-vllm-docker stays conflict-free.

### Tune + deploy

```bash
# 1. Tune your model
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2

# 2. Sync configs to the mod (the symlink makes them available to spark-vllm-docker)
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --sync-mod

# 3. Launch with the mod
cd ~/spark-vllm-docker
./launch-cluster.sh --apply-mod mods/vllm-tune exec vllm serve ...
```

### In a recipe YAML

```yaml
# In your spark-vllm-docker recipe:
mods:
  - mods/vllm-tune
  - mods/fix-qwen3-coder-next
```

> **Tip:** You can also use `--apply-mod` with an absolute path:
> `--apply-mod ~/code/vllm-tune/mod` (no symlink needed).

## Contributing Configs

Tuned configs are model- and hardware-specific — the more we collect, the fewer
people need to run hours-long tuning jobs. **PRs with new configs are welcome!**

To contribute your tuned configs:

1. Run tuning for your model:
   ```bash
   ./vllm-tune.sh <YOUR_MODEL> --tp <N>
   ```
2. Verify the configs work (deploy + benchmark)
3. Submit a PR adding the `configs/<model-slug>/tp<N>/` directory with:
   - `moe/*.json` — MoE kernel configs
   - `fp8/*.json` — FP8 dense GEMM configs
   - `metadata.json` — auto-generated provenance (model, TP, vLLM version, host, timestamp)

The directory structure is self-describing — just drop your tuned files in and
open a PR.

## License

Apache 2.0 — see [LICENSE](LICENSE).

