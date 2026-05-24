# FP8 Dense GEMM Tuning Report

Generated: 2026-05-14T01:59:14-05:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` |
| TP size | 1 |
| Container | vllm_node |
| Total time | 110m 19s |

## Results

| Shape (N,K) | Status | Time |
|--------------------|--------|------|
| 704,2816 | ✅ OK | 33m 11s |
| 2112,2816 | ✅ OK | 6m 26s |
| 2816,704 | ✅ OK | 2m 10s |
| 2816,2112 | ✅ OK | 5m 44s |
| 2816,4096 | ✅ OK | 41m 9s |
| 8192,2816 | ✅ OK | 21m 39s |

## Summary

- **Succeeded:** 6/6
- **Failed:** 0/6
