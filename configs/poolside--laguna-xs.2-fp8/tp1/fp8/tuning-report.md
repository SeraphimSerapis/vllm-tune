# FP8 Dense GEMM Tuning Report

Generated: 2026-05-21T01:35:10-05:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `poolside/Laguna-XS.2-FP8` |
| TP size | 1 |
| Container | vllm_node |
| Total time | 47m 14s |

## Results

| Shape (N,K) | Status | Time |
|--------------------|--------|------|
| 512,2048 | ✅ OK | 1m 52s |
| 2048,512 | ✅ OK | 1m 31s |
| 2048,6144 | ✅ OK | 12m 3s |
| 2048,8192 | ✅ OK | 16m 3s |
| 8192,2048 | ✅ OK | 15m 45s |

## Summary

- **Succeeded:** 5/5
- **Failed:** 0/5
