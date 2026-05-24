# MoE Tuning Report

Generated: 2026-05-17T22:07:17-05:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` |
| TP size | 1 |
| Container | vllm_node |
| Total time | 2755m 5s |

## Results

| Batch Size | Status | Time |
|--------------------|--------|------|
| 1 | ✅ OK | 47m 30s |
| 2 | ✅ OK | 110m 55s |
| 4 | ✅ OK | 43m 53s |
| 8 | ✅ OK | 72m 45s |
| 16 | ✅ OK | 107m 53s |
| 24 | ✅ OK | 130m 6s |
| 32 | ✅ OK | 145m 11s |
| 48 | ✅ OK | 157m 6s |
| 64 | ✅ OK | 161m 15s |
| 96 | ✅ OK | 170m 42s |
| 128 | ✅ OK | 166m 40s |
| 256 | ✅ OK | 169m 0s |
| 512 | ✅ OK | 174m 10s |
| 1024 | ✅ OK | 176m 35s |
| 1536 | ✅ OK | 182m 11s |
| 2048 | ✅ OK | 197m 40s |
| 3072 | ✅ OK | 229m 49s |
| 4096 | ✅ OK | 311m 42s |

## Summary

- **Succeeded:** 18/18
- **Failed:** 0/18
