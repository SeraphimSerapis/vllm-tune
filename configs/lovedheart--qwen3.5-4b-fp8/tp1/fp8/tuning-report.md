# FP8 Dense GEMM Tuning Report

Generated: 2026-05-19T02:15:17-05:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `lovedheart/Qwen3.5-4B-FP8` |
| TP size | 1 |
| Container | vllm_node |
| Total time | 89m 9s |

## Results

| Shape (N,K) | Status | Time |
|--------------------|--------|------|
| 2560,4096 | ✅ OK | 10m 6s |
| 2560,9216 | ✅ OK | 22m 14s |
| 6144,2560 | ✅ OK | 14m 46s |
| 8192,2560 | ✅ OK | 19m 42s |
| 9216,2560 | ✅ OK | 22m 21s |

## Summary

- **Succeeded:** 5/5
- **Failed:** 0/5
