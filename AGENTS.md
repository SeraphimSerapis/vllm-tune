# vLLM-Tune — Agent Guide

This is a CLI tool for tuning vLLM's Triton kernel configs on NVIDIA GPUs.
It benchmarks MoE and FP8 GEMM kernels inside a running Docker container,
saves optimal configs as JSON, and deploys them via `docker cp`.

---

## Architecture

```
vllm-tune.sh                ← Main entry point (orchestrator)
├── tune-moe.sh             ← MoE fused expert dispatch tuning
├── tune-fp8.sh             ← FP8 block-scaled GEMM tuning
├── dist-tune.py            ← Distributed tuning across cluster nodes
├── lib/common.sh           ← Shared utilities (sourced by tune-*.sh)
├── lib/detect.py           ← Centralized model arch & shape detection
├── mod/run.sh              ← Container mod for persistent deployment
└── configs/                ← Config store (pre-shipped + user-generated)
    └── <model-slug>/tp<N>/
        ├── moe/*.json      ← E=...,N=... kernel configs
        ├── fp8/*.json      ← N=...,K=... kernel configs
        └── metadata.json   ← Provenance (model, TP, vLLM version, timestamp)
```

### Data flow

```
1. vllm-tune.sh parses args → delegates to tune-moe.sh / tune-fp8.sh
2. tune-*.sh runs benchmarks inside Docker container (via docker exec)
3. After each batch/shape: docker cp results out → jq merge into configs/
4. On completion: optional docker cp deploy into container's vLLM site-packages
```

### Key design decisions

- **Crash-safe**: After every single batch size (MoE) or shape (FP8), results
  are copied from the container and merged into host-side backup. A power loss
  mid-tuning loses only the current item.
- **Incremental**: `jq -s '.[0] * .[1]'` merging means re-running with a
  subset of batch sizes extends (not overwrites) the config.
- **No container modifications**: All tuning commands use `docker exec`. The
  container image is never modified.

---

## File responsibilities

| File | Role | Modifiable in isolation? |
|------|------|-------------------------|
| `vllm-tune.sh` | CLI parsing, tmux, deploy, mod sync, arch detection, metadata | Yes |
| `tune-moe.sh` | MoE benchmark loop (calls `benchmark_moe.py` via git sparse-checkout) | Yes |
| `tune-fp8.sh` | FP8 benchmark loop (Triton autotuning benchmark) | Yes |
| `dist-tune.py` | Distributed orchestration (Rich dashboard, worker pool, rsync) | Yes |
| `lib/common.sh` | Shared functions (MUST be updated if changing retry/merge/report behavior) | Careful — sourced by both tune scripts |
| `lib/detect.py` | Centralized MoE/dense detection + FP8 shape detection | Yes — consumed by vllm-tune.sh, tune-fp8.sh, dist-tune.py |
| `mod/run.sh` | Runs inside container at startup — copies JSON configs into vLLM paths | Yes |
| `tests/test-vllm-tune.sh` | Offline test suite (no Docker/GPU required) | Yes |
| `configs/` | JSON kernel configs — tracked in git, never auto-deleted | Append-only |

---

## Conventions

### Script conventions

- All scripts use `set -euo pipefail`
- Error output goes to stderr via `die()` / `echo ... >&2`
- Functions are documented with comments specifying arguments and globals
- ANSI colors used for terminal output (bold, cyan, red for errors)
- Environment variables override defaults (never hardcode user-specific values)

### Config conventions

- Config filenames are set by vLLM's benchmarking tools — do NOT rename them
- MoE format: `E=<experts>,N=<dim>,device_name=<GPU>,dtype=<dtype>,block_shape=[128,128].json`
- FP8 format: `N=<out>,K=<in>,device_name=<GPU>,dtype=fp8_w8a8,block_shape=[128,128].json`
- Model slug: lowercase, `/` → `--`, special chars → `-` (e.g. `qwen--qwen3.6-35b-a3b-fp8`)
- Config store path: `configs/<model-slug>/tp<N>/{moe,fp8}/`

### Naming

- The project is **vLLM-Tune** (capital L, capital T, hyphen)
- The CLI command is **`vllm-tune.sh`** (lowercase, hyphen)
- The mod is **`vllm-tune`** (lowercase, hyphen)

---

## Common tasks

### Architecture detection (MoE vs dense)

All model detection logic is centralized in `lib/detect.py`. This single
Python module is `cat`-piped into the container via `docker exec` and outputs
structured JSON. It is consumed by `vllm-tune.sh` (arch detection),
`tune-fp8.sh` (shape detection), and `dist-tune.py` (both).

Detection checks both `num_local_experts` (standard) and `n_routed_experts`
(DeepSeek-V3/V4) to identify MoE models. Shape detection covers QKV
projections, attention output, linear attention (Mamba), shared experts, dense
FFN, and MoE expert FFN (`moe_intermediate_size` for DeepSeek).

- **`--mode all` + dense model**: MoE phase is skipped with informative message,
  FP8 tuning proceeds normally
- **`--mode moe` + dense model**: Clear error with suggestion to use `--mode fp8`
- **`--dry-run`**: Detection is skipped (no container needed)
- **Detection failure**: Falls back to `IS_MOE=false` (safe default — skips MoE)

The gating logic lives in `vllm-tune.sh` near the `# Architecture detection`
comment block. It does NOT modify `tune-moe.sh` — the gating is an
orchestration concern.

**Rule**: If you add new model config attributes to detect, update
`lib/detect.py` — the consumers (`vllm-tune.sh`, `tune-fp8.sh`, `dist-tune.py`)
parse its JSON output and don't need changes.

### Distributed tuning (`--dist`)

`vllm-tune.sh --dist` delegates to `dist-tune.py`, which orchestrates tuning
across multiple cluster nodes.

```
1. Discover nodes (--nodes flag, CLUSTER_NODES env, or .env file)
2. Detect arch + shapes via lib/detect.py (on head node's container)
3. Build task queue: MoE batch sizes + FP8 shapes
4. Scan local configs → skip already-tuned items (resume support)
5. rsync repo to remote nodes
6. Dispatch tasks: each node runs vllm-tune.sh for one item at a time
7. After each task: rsync configs back → deep-merge into head node
8. Rich TUI dashboard shows live status per node
```

- **Node discovery**: `--nodes` flag > `CLUSTER_NODES` env > `~/spark-vllm-docker/.env`
- **Resume**: Scans JSON config files for existing batch size keys
- **Merge**: Deep merge matching `jq -s '.[0] * .[1]'` semantics
- **Cleanup**: Ctrl-C kills remote tuning processes via SSH
- **Dependency**: Requires `pip install rich` on the head node

### Adding support for a new tuning mode

If vLLM adds a new kernel type that benefits from tuning:

1. Create `tune-<type>.sh` following `tune-moe.sh` as template
2. Source `lib/common.sh` for shared utilities
3. Add a new `--mode <type>` option to `vllm-tune.sh`
4. Add a `CONFIGS_<TYPE>` path and deploy target in `deploy_all()`
5. Update `mod/run.sh` to install the new config type
6. Add tests to `tests/test-vllm-tune.sh`

### Adding pre-shipped configs for a new model

1. Run tuning: `./vllm-tune.sh <model> --tp <N>`
2. Verify results with a benchmark
3. Commit the `configs/<model-slug>/tp<N>/` directory
4. Add a row to the "Pre-shipped Configs" table in `README.md`

### Modifying shared behavior (retry, merge, reports)

All shared logic lives in `lib/common.sh`. Both `tune-moe.sh` and `tune-fp8.sh`
source it. Key functions:

| Function | What it does |
|----------|-------------|
| `preflight()` | Container check + tini detection + jq check |
| `run_with_retry()` | Execute command with MAX_RETRIES retries |
| `merge_results()` | Crash-safe jq merge from container to host |
| `post_round()` | Clear caches + check for zombies |
| `generate_report()` | Write Markdown tuning report |
| `print_summary()` | Print styled terminal summary |
| `cleanup_zombies()` | Restart container if zombie processes accumulated |
| `clear_caches()` | Drop OS page caches (configurable via env vars) |

**Rule**: If you change a function signature in `common.sh`, you must update
both `tune-moe.sh` and `tune-fp8.sh`.

### Modifying the container deployment paths

vLLM kernel config paths are auto-detected at runtime:

```bash
# In vllm-tune.sh:
detect_vllm_paths()  # runs: python3 -c "import vllm; ..."

# In mod/run.sh:
VLLM_ROOT=$(python3 -c "import vllm, pathlib; ...")
```

Both fall back to `/usr/local/lib/python3.12/dist-packages/vllm` if detection
fails. If vLLM changes its internal config directory structure, update:

1. `detect_vllm_paths()` in `vllm-tune.sh`
2. The install targets in `mod/run.sh`
3. The path documentation in `README.md` (Deployment section)

---

## What NOT to do

- **Don't rename config JSON files** — vLLM matches them by exact filename
  pattern (device_name, dtype, block_shape are all parsed from the filename)
- **Don't merge configs across TP sizes** — N is divided by TP, so the shapes
  are fundamentally different
- **Don't remove the `metadata.json` writing** — it's the only provenance
  record (vLLM version, hostname, timestamp)
- **Don't hardcode personal paths** — use environment variables
- **Don't modify pre-shipped configs** — they're community-contributed and
  should only be updated by re-running tuning with a newer vLLM version

---

## Testing

### Automated test suite

```bash
# Run the full offline test suite (no Docker/GPU required):
./tests/test-vllm-tune.sh
```

The test suite covers:
- Version and help output
- Argument validation (missing model, invalid mode, unknown flags)
- Model slug generation
- Dry-run flow (mode selection, phase headers, flag passthrough)
- Config path construction
- Architecture detection gating (lib/detect.py presence, detection attributes)
- DeepSeek-V3/V4 support (n_routed_experts, moe_intermediate_size, V4 sed patch)
- Distributed tuning (`--dist` flag, `dist-tune.py` presence)
- Script syntax validation (all `.sh` files + Python files)
- Documentation checks (README + AGENTS.md)

Tests exit with code 0 on success, non-zero on failure.

### Manual testing

```bash
# Dry-run (no container needed for basic flow):
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --dry-run --tp 2

# Version check:
./vllm-tune.sh --version

# Deploy-only (requires running container):
./vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy-only --dry-run

# Help:
./vllm-tune.sh --help
./tune-moe.sh --help
./tune-fp8.sh --help
```

---

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `VLLM_TUNE_HOME` | vllm-tune.sh | Override config store root |
| `VLLM_TUNE_CONFIG` | lib/common.sh | Override config.json location |
| `CONTAINER` | all scripts | Docker container name (default: `vllm_node`) |
| `CONFIGS_DIR` | tune-*.sh | Override config output directory |
| `HOST_BACKUP_DIR` | tune-*.sh | Incremental backup directory |
| `TUNE_SCRIPTS_DIR` | vllm-tune.sh | Override location of tune-*.sh scripts |
| `SPARKRUN_TUNING_DIR` | vllm-tune.sh | sparkrun tuning cache path |
| `PEER_NODES` | lib/common.sh | SSH peers for cache clearing |
| `DROP_CACHES_CMD` | lib/common.sh | Custom cache-drop command |
| `MAX_RETRIES` | tune-*.sh | Retry count per tuning item (default: 1) |
| `CLUSTER_NODES` | dist-tune.py | Comma-separated node IPs for `--dist` |

---

## Integration with spark-vllm-docker

This tool is designed to work with the `spark-vllm-docker` container
infrastructure but does **not** depend on it. The only integration point is the
`--sync-mod` workflow.

### Mod lifecycle

```
mod/run.sh (repo)                    ← Source of truth
    │
    ├── --sync-mod (first run)       → Creates ~/scripts/vllm/recipes/mods/vllm-tune/
    │                                   ├── run.sh        (copied from mod/run.sh)
    │                                   ├── configs/      (MoE JSON files)
    │                                   └── fp8-configs/  (FP8 JSON files)
    │
    ├── --sync-mod (subsequent)      → Updates run.sh + syncs new JSON configs
    │
    └── launch-cluster.sh --apply-mod → Executes run.sh inside container at startup
                                        → Copies JSON configs into vLLM site-packages
```

`--sync-mod` auto-bootstraps: if the mod directory doesn't exist, it creates
it from the bundled `mod/run.sh`. On every sync, `run.sh` is updated to match
the repo version.

### Without spark-vllm-docker

For standalone Docker setups, `--deploy` / `--deploy-only` uses `docker cp`
directly — no mod infrastructure needed.

