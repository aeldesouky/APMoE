#!/usr/bin/env python3
"""APMoE Concurrent Load Test — GET /health endpoint.

Measures RPS, avg latency, p50, p95, p99, and max latency under N concurrent
virtual users sending sequential requests over a fixed duration.

Usage:
    python scripts/load_test.py [--url URL] [--users N] [--duration SECS]
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Result:
    status: int
    latency_ms: float


@dataclass
class Stats:
    total: int = 0
    ok: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def rps(self) -> float:
        return self.total / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.95)]

    @property
    def p99_ms(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.99)]

    @property
    def max_ms(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def error_rate_pct(self) -> float:
        return (self.errors / self.total * 100) if self.total else 0.0


async def worker(url: str, duration: float, results: list[Result]) -> None:
    """One virtual user: send GET /health in a tight loop for *duration* seconds."""
    import httpx

    deadline = time.monotonic() + duration
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            try:
                resp = await client.get(url)
                latency_ms = (time.monotonic() - t0) * 1000
                results.append(Result(status=resp.status_code, latency_ms=latency_ms))
            except Exception:
                latency_ms = (time.monotonic() - t0) * 1000
                results.append(Result(status=0, latency_ms=latency_ms))


async def run_load_test(url: str, users: int, duration: float) -> Stats:
    results: list[Result] = []
    t_start = time.monotonic()
    await asyncio.gather(*[worker(url, duration, results) for _ in range(users)])
    wall = time.monotonic() - t_start

    stats = Stats(wall_seconds=wall)
    for r in results:
        stats.total += 1
        if 200 <= r.status < 300:
            stats.ok += 1
            stats.latencies.append(r.latency_ms)
        else:
            stats.errors += 1
            stats.latencies.append(r.latency_ms)  # include error latencies
    return stats


def print_report(stats: Stats, url: str, users: int, duration: float) -> None:
    print("\n" + "=" * 60)
    print("  APMoE Load Test Results")
    print("=" * 60)
    print(f"  Target URL      : {url}")
    print(f"  Virtual users   : {users}")
    print(f"  Duration        : {duration}s")
    print(f"  Wall time       : {stats.wall_seconds:.2f}s")
    print("-" * 60)
    print(f"  Total requests  : {stats.total}")
    print(f"  Successful (2xx): {stats.ok}")
    print(f"  Errors          : {stats.errors}")
    print(f"  Error rate      : {stats.error_rate_pct:.1f}%")
    print("-" * 60)
    print(f"  RPS (req/s)     : {stats.rps:.1f}")
    print(f"  Avg latency     : {stats.avg_ms:.1f} ms")
    print(f"  p50 latency     : {stats.p50_ms:.1f} ms")
    print(f"  p95 latency     : {stats.p95_ms:.1f} ms")
    print(f"  p99 latency     : {stats.p99_ms:.1f} ms")
    print(f"  Max latency     : {stats.max_ms:.1f} ms")
    print("=" * 60)

    # Simple pass/fail verdict for the checklist
    ok = stats.error_rate_pct == 0.0
    print(f"\n  Verdict: {'✅ PASS — 0% error rate' if ok else '❌ FAIL — errors detected'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="APMoE concurrent load test")
    parser.add_argument("--url", default="http://127.0.0.1:8000/health",
                        help="Endpoint to hit (default: GET /health)")
    parser.add_argument("--users", type=int, default=20,
                        help="Number of concurrent virtual users (default: 20)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Test duration in seconds (default: 15)")
    args = parser.parse_args()

    print(f"\nStarting load test: {args.users} users × {args.duration}s → {args.url}")
    stats = asyncio.run(run_load_test(args.url, args.users, args.duration))
    print_report(stats, args.url, args.users, args.duration)
    sys.exit(0 if stats.error_rate_pct == 0.0 else 1)


if __name__ == "__main__":
    main()
