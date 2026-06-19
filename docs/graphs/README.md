# Performance Testing Graphs

This directory contains graphs generated from the local APMoE performance
benchmarks for Chapter 6.

## Graphs

| File | Description |
|---|---|
| `throughput_vs_concurrency.png` | Requests per second for health check, keystroke-only, image-only, and multimodal prediction at 2, 10, and 25 concurrent users. |
| `average_latency_vs_concurrency.png` | Average response latency across the same concurrency levels and request types. |
| `p95_latency_vs_concurrency.png` | p95 latency trend, useful for discussing tail latency under load. |
| `latency_percentiles_25_users.png` | p50, p95, p99, and maximum latency comparison at 25 concurrent users. |
| `performance_dashboard.png` | Combined 2x2 dashboard containing all four graphs. |

## Benchmark Context

The graphs are based on 10-second local load tests with short warm-up phases.
The benchmark covered health checks, keystroke-only prediction, image-only
prediction, and full multimodal prediction. The local environment used the
available CPU PyTorch face model artifact for facial inference.

These results are useful for thesis evaluation and prototype comparison, but
they should not be treated as production SLA guarantees without repeating the
tests on the final deployment hardware and runtime stack.
