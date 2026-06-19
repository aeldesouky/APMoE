#!/usr/bin/env python3
"""APMoE multimodal load test for /predict.

This script benchmarks the FastAPI prediction route in-process through
httpx.ASGITransport. It avoids depending on a running uvicorn server while still
exercising request serialization, middleware, routing, async prediction, modality
processing, expert inference, and aggregation.

Examples:
    python scripts/load_test_multimodal.py --mode all
    python scripts/load_test_multimodal.py --mode multimodal --users 2 --duration 10
    python scripts/load_test_multimodal.py --face-weights src/apmoe/weights/mobilenet_age_model.pth.zip
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Mode = Literal["health", "keystroke", "image", "multimodal"]


KEYSTROKE_SESSION = [
    [65, 83, 120],
    [83, 68, 98],
    [68, 70, 110],
    [70, 71, 95],
    [71, 72, 88],
    [72, 74, 102],
    [74, 75, 115],
    [75, 76, 93],
    [76, 59, 107],
    [59, 65, 119],
    [65, 83, 126],
    [83, 68, 91],
    [68, 70, 105],
    [70, 71, 118],
    [71, 72, 88],
    [72, 74, 100],
    [74, 75, 112],
    [75, 76, 97],
    [76, 59, 108],
    [59, 65, 122],
]


@dataclass
class Stats:
    """Aggregated load-test statistics."""

    mode: Mode
    users: int
    duration_s: float
    total: int = 0
    ok: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def rps(self) -> float:
        return self.total / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def error_rate_pct(self) -> float:
        return self.errors / self.total * 100 if self.total else 0.0

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    def percentile_ms(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        values = sorted(self.latencies_ms)
        index = min(len(values) - 1, int(len(values) * pct))
        return values[index]

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "users": self.users,
            "duration_s": self.duration_s,
            "wall_seconds": round(self.wall_seconds, 3),
            "total_requests": self.total,
            "successful_requests": self.ok,
            "errors": self.errors,
            "error_rate_pct": round(self.error_rate_pct, 3),
            "rps": round(self.rps, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.percentile_ms(0.95), 3),
            "p99_ms": round(self.percentile_ms(0.99), 3),
            "max_ms": round(self.max_ms, 3),
        }


def _make_image_b64(size: int = 96) -> str:
    """Create a deterministic PNG and return it as base64 text."""
    from PIL import Image

    image = Image.new("RGB", (size, size), (120, 150, 180))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, ((x + y) * 2) % 256)

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _payload_for(mode: Mode, image_b64: str) -> dict[str, Any] | None:
    if mode == "health":
        return None
    if mode == "keystroke":
        return {"keystroke": KEYSTROKE_SESSION}
    if mode == "image":
        return {"image": image_b64}
    if mode == "multimodal":
        return {"keystroke": KEYSTROKE_SESSION, "image": image_b64}
    raise ValueError(f"Unsupported mode: {mode}")


def _config_with_face_override(config_path: Path, face_weights: str | None) -> Path:
    """Return a config path, optionally with the face expert weights replaced."""
    if face_weights is None:
        return config_path

    config = json.loads(config_path.read_text(encoding="utf-8"))
    for expert in config["apmoe"]["experts"]:
        if expert.get("name") == "face_age_expert":
            expert["weights"] = face_weights
            break
    else:
        raise ValueError("No face_age_expert entry found in config.")

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="apmoe-load-test-",
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        json.dump(config, tmp, indent=2)
    return Path(tmp.name)


async def _run_one_mode(
    *,
    api: Any,
    mode: Mode,
    users: int,
    duration_s: float,
    warmup: int,
    image_b64: str,
) -> Stats:
    import httpx

    transport = httpx.ASGITransport(app=api)
    payload = _payload_for(mode, image_b64)
    endpoint = "/v1/health" if mode == "health" else "/v1/predict"

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        timeout=120.0,
    ) as client:
        for _ in range(warmup):
            if mode == "health":
                await client.get(endpoint)
            else:
                await client.post(endpoint, json=payload)

        results: list[tuple[int, float]] = []

        async def worker() -> None:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                started = time.perf_counter()
                try:
                    if mode == "health":
                        response = await client.get(endpoint)
                    else:
                        response = await client.post(endpoint, json=payload)
                    latency_ms = (time.perf_counter() - started) * 1000
                    results.append((response.status_code, latency_ms))
                except Exception:
                    latency_ms = (time.perf_counter() - started) * 1000
                    results.append((0, latency_ms))

        started = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(users)])
        wall_seconds = time.perf_counter() - started

    stats = Stats(mode=mode, users=users, duration_s=duration_s, wall_seconds=wall_seconds)
    for status, latency_ms in results:
        stats.total += 1
        stats.latencies_ms.append(latency_ms)
        if 200 <= status < 300:
            stats.ok += 1
        else:
            stats.errors += 1
    return stats


def _print_table(rows: list[Stats]) -> None:
    headers = [
        "Mode",
        "Users",
        "Total",
        "RPS",
        "Err%",
        "Avg ms",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "Max ms",
    ]
    table_rows = [
        [
            row.mode,
            str(row.users),
            str(row.total),
            f"{row.rps:.2f}",
            f"{row.error_rate_pct:.2f}",
            f"{row.avg_ms:.2f}",
            f"{row.p50_ms:.2f}",
            f"{row.percentile_ms(0.95):.2f}",
            f"{row.percentile_ms(0.99):.2f}",
            f"{row.max_ms:.2f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]
    print("\nAPMoE multimodal load-test results")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


async def _amain(args: argparse.Namespace) -> None:
    from apmoe.core.app import APMoEApp
    from apmoe.serving.app_factory import create_api

    logging.getLogger("apmoe.pipeline").setLevel(logging.ERROR)

    config_path = _config_with_face_override(Path(args.config), args.face_weights)
    app = APMoEApp.from_config(config_path)
    app.config.apmoe.serving.authentication_enabled = False
    app.config.apmoe.serving.authorization_enabled = False
    api = create_api(app)

    image_b64 = _make_image_b64(args.image_size)
    if args.mode == "all":
        modes: list[Mode] = ["health", "keystroke", "image", "multimodal"]
    else:
        modes = [args.mode]

    rows: list[Stats] = []
    for mode in modes:
        rows.append(
            await _run_one_mode(
                api=api,
                mode=mode,
                users=args.users,
                duration_s=args.duration,
                warmup=args.warmup,
                image_b64=image_b64,
            )
        )

    _print_table(rows)
    if args.json:
        Path(args.json).write_text(
            json.dumps([row.as_dict() for row in rows], indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report written to {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="APMoE multimodal /predict load test")
    parser.add_argument("--config", default="configs/multimodal.json")
    parser.add_argument(
        "--face-weights",
        default=None,
        help="Optional replacement weights path for face_age_expert.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "health", "keystroke", "image", "multimodal"],
        default="all",
    )
    parser.add_argument("--users", type=int, default=2)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--json", default=None, help="Optional JSON report output path.")
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
