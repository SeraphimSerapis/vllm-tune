# FP8 Dense GEMM Tuning Report

Generated: 2026-05-19T20:08:57-05:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` |
| TP size | 1 |
| Container | vllm_node |
| Total time | 166m 57s |

## Results

| Shape (N,K) | Status | Time |
|--------------------|--------|------|
| 3136,5120 | ✅ OK | 36m 6s |
| 3136,12544 | ✅ OK | 69m 3s |
| 7168,3136 | ✅ OK | 21m 8s |
| 12544,3136 | ✅ OK | 40m 39s |

## Summary

- **Succeeded:** 4/4
- **Failed:** 0/4
