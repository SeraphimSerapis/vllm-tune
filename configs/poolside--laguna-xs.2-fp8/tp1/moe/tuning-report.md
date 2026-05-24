# MoE Tuning Report

Generated: 2026-05-22T09:54:02-05:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `poolside/Laguna-XS.2-FP8` |
| TP size | 1 |
| Container | vllm_node |
| Total time | 1307m 15s |

## Results

| Batch Size | Status | Time |
|--------------------|--------|------|
| 1 | ✅ OK | 11m 13s |
| 2 | ✅ OK | 13m 23s |
| 4 | ✅ OK | 13m 0s |
| 8 | ✅ OK | 25m 10s |
| 16 | ✅ OK | 46m 58s |
| 24 | ✅ OK | 53m 19s |
| 32 | ✅ OK | 64m 38s |
| 48 | ✅ OK | 78m 17s |
| 64 | ✅ OK | 88m 0s |
| 96 | ✅ OK | 92m 47s |
| 128 | ✅ OK | 98m 16s |
| 256 | ✅ OK | 92m 21s |
| 512 | ✅ OK | 97m 13s |
| 1024 | ✅ OK | 97m 3s |
| 1536 | ✅ OK | 104m 2s |
| 2048 | ✅ OK | 105m 34s |
| 3072 | ✅ OK | 107m 32s |
| 4096 | ✅ OK | 118m 28s |

## Summary

- **Succeeded:** 18/18
- **Failed:** 0/18
