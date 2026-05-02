#!/usr/bin/env python3
"""APMoE /predict load test — sends real keystroke inference requests concurrently."""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

PREDICT_PAYLOAD = {
    "keystroke": [
        [65, 83, 120], [83, 68, 98], [68, 70, 110], [70, 71, 95],
        [71, 72, 88], [72, 74, 102], [74, 75, 115], [75, 76, 93],
        [76, 59, 107], [59, 65, 119], [65, 83, 126], [83, 68, 91],
        [68, 70, 105], [70, 71, 118], [71, 72, 88], [72, 74, 100],
        [74, 75, 112], [75, 76, 97], [76, 59, 108], [59, 65, 122],
    ]
}

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


async def worker(url: str, duration: float, results: list[tuple[int, float]]) -> None:
    import httpx
    deadline = time.monotonic() + duration
    async with httpx.AsyncClient(timeout=30.0) as client:
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            try:
                resp = await client.post(url, json=PREDICT_PAYLOAD)
                latency_ms = (time.monotonic() - t0) * 1000
                results.append((resp.status_code, latency_ms))
            except Exception:
                results.append((0, (time.monotonic() - t0) * 1000))


async def run(url: str, users: int, duration: float) -> Stats:
    results: list[tuple[int, float]] = []
    t0 = time.monotonic()
    await asyncio.gather(*[worker(url, duration, results) for _ in range(users)])
    wall = time.monotonic() - t0
    stats = Stats(wall_seconds=wall)
    for status, lat in results:
        stats.total += 1
        if 200 <= status < 300:
            stats.ok += 1
            stats.latencies.append(lat)
        else:
            stats.errors += 1
            stats.latencies.append(lat)
    return stats


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/predict"
    users = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

    print(f"\nStarting /predict load test: {users} users × {duration}s → {url}")
    stats = asyncio.run(run(url, users, duration))

    avg = statistics.mean(stats.latencies) if stats.latencies else 0.0
    p50 = statistics.median(stats.latencies) if stats.latencies else 0.0
    print("\n" + "=" * 60)
    print("  APMoE /predict Load Test Results")
    print("=" * 60)
    print(f"  Virtual users   : {users}")
    print(f"  Duration        : {duration}s  |  Wall: {stats.wall_seconds:.2f}s")
    print(f"  Total requests  : {stats.total}")
    print(f"  Successful (2xx): {stats.ok}")
    print(f"  Errors          : {stats.errors}  ({stats.total and stats.errors/stats.total*100:.1f}%)")
    print("-" * 60)
    print(f"  RPS             : {stats.rps:.1f}")
    print(f"  Avg latency     : {avg:.0f} ms")
    print(f"  p50 latency     : {p50:.0f} ms")
    print(f"  p95 latency     : {stats.p95_ms:.0f} ms")
    print(f"  p99 latency     : {stats.p99_ms:.0f} ms")
    print(f"  Max latency     : {max(stats.latencies, default=0):.0f} ms")
    print("=" * 60)
    ok = stats.errors == 0
    print(f"\n  Verdict: {'✅ PASS' if ok else '❌ FAIL'}\n")


if __name__ == "__main__":
    main()
