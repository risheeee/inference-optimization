"""
load_generator.py — Async Concurrent Load Generator
=====================================================
Simulates heavy, concurrent user traffic against an OpenAI-compatible
/v1/completions or /v1/chat/completions endpoint (vLLM or llama.cpp).

Measures per-request:
  • TTFT  — Time To First Token (ms), measured via SSE streaming
  • Total Latency (ms)
  • Tokens Generated (approximate from SSE chunks)
  • Tokens/Second (throughput per request)

Writes a JSONL results file to results/{engine}_{profile}_{timestamp}.jsonl

Usage:
  python src/load_generator.py run --engine vllm --profile short_qa
  python src/load_generator.py run --engine llamacpp --profile medium_reasoning --concurrency 8
  python src/load_generator.py mock  # runs against a local mock echo server for testing
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
REQUEST_TIMEOUT_S = 120

ENGINE_DEFAULTS = {
    "vllm": {
        "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
        "model": os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
    },
    "llamacpp": {
        "base_url": os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080"),
        "model": os.getenv("LLAMACPP_MODEL_NAME", "llama-3-8b-instruct"),
    },
    "mock": {
        "base_url": "http://localhost:9999",
        "model": "mock-model",
    },
}

console = Console()
app = typer.Typer(
    name="load-generator",
    help="Async concurrent load generator for LLM inference benchmarking.",
    add_completion=False,
)


# ══════════════════════════════════════════════════════════════════════════════
# Core Request Logic
# ══════════════════════════════════════════════════════════════════════════════

async def send_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    request_id: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Sends a single streaming /v1/completions request and records timing metrics.

    Returns a result dict with TTFT, total latency, tokens generated, etc.
    """
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": DEFAULT_TEMPERATURE,
    }

    start_wall = time.perf_counter()
    ttft_s: Optional[float] = None
    tokens_generated = 0
    full_response = ""
    error_msg: Optional[str] = None
    status_code: Optional[int] = None

    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            status_code = resp.status
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        if text:
                            if ttft_s is None:
                                ttft_s = time.perf_counter() - start_wall
                            full_response += text
                            tokens_generated += 1  # 1 SSE chunk ≈ 1 token (engine-dependent)
                except json.JSONDecodeError:
                    pass

    except asyncio.TimeoutError:
        error_msg = f"Request timed out after {REQUEST_TIMEOUT_S}s"
    except Exception as exc:
        error_msg = str(exc)

    total_latency_s = time.perf_counter() - start_wall

    return {
        "request_id": request_id,
        "run_id": None,  # filled by caller
        "prompt_preview": prompt[:120],
        "prompt_chars": len(prompt),
        "ttft_ms": round(ttft_s * 1000, 2) if ttft_s else None,
        "total_latency_ms": round(total_latency_s * 1000, 2),
        "tokens_generated": tokens_generated,
        "tokens_per_second": round(tokens_generated / total_latency_s, 2) if total_latency_s > 0 else 0,
        "response_preview": full_response[:200],
        "full_response": full_response,
        "http_status": status_code,
        "status": "error" if error_msg else "success",
        "error": error_msg,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Load Test Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class LoadTestRunner:
    def __init__(
        self,
        base_url: str,
        model: str,
        engine_label: str,
        profile_label: str,
    ):
        self.base_url = base_url
        self.model = model
        self.engine_label = engine_label
        self.profile_label = profile_label
        self.results: list[dict] = []
        self.run_id = str(uuid.uuid4())[:8]

    async def run(
        self,
        prompts: list[str],
        total_requests: int,
        concurrency: int,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[dict]:
        """Execute the load test with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        all_prompts = [prompts[i % len(prompts)] for i in range(total_requests)]

        # Live stats tracking
        stats = {
            "completed": 0, "errors": 0,
            "total_ttft_ms": 0.0, "total_latency_ms": 0.0, "total_tps": 0.0,
        }
        start_time = time.perf_counter()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        connector = aiohttp.TCPConnector(limit=concurrency + 20, limit_per_host=concurrency + 20)

        async def bounded_request(session: aiohttp.ClientSession, prompt: str, req_id: int):
            async with semaphore:
                result = await send_streaming_request(session, self.base_url, self.model, prompt, req_id, max_tokens)
                result["run_id"] = self.run_id
                return result

        with progress:
            pt = progress.add_task(
                f"[cyan]{self.engine_label}/{self.profile_label}",
                total=total_requests,
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [
                    asyncio.create_task(bounded_request(session, p, i))
                    for i, p in enumerate(all_prompts)
                ]
                self.results = []
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    self.results.append(result)

                    # Update live stats
                    if result["status"] == "success":
                        stats["completed"] += 1
                        if result["ttft_ms"]:
                            stats["total_ttft_ms"] += result["ttft_ms"]
                        stats["total_latency_ms"] += result["total_latency_ms"]
                        stats["total_tps"] += result["tokens_per_second"]
                    else:
                        stats["errors"] += 1

                    progress.advance(pt)

        elapsed = time.perf_counter() - start_time
        success_count = stats["completed"]

        # ── Print summary table ───────────────────────────────────────────────
        table = Table(title=f"📊 Load Test Results — {self.engine_label} / {self.profile_label}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", min_width=28)
        table.add_column("Value", style="green", justify="right")

        latencies = sorted([r["total_latency_ms"] for r in self.results if r["status"] == "success"])
        p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

        table.add_row("Total Requests", str(total_requests))
        table.add_row("Successful", str(success_count))
        table.add_row("Errors", str(stats["errors"]))
        table.add_row("Concurrency", str(concurrency))
        table.add_row("Total Duration (s)", f"{elapsed:.2f}")
        table.add_row("Requests/Second (RPS)", f"{total_requests / elapsed:.2f}")
        if success_count > 0:
            table.add_row("Avg TTFT (ms)", f"{stats['total_ttft_ms'] / success_count:.1f}")
            table.add_row("Avg Total Latency (ms)", f"{stats['total_latency_ms'] / success_count:.1f}")
            table.add_row("P50 Latency (ms)", f"{p50:.1f}")
            table.add_row("P95 Latency (ms)", f"{p95:.1f}")
            table.add_row("P99 Latency (ms)", f"{p99:.1f}")
            table.add_row("Avg Tokens/Second", f"{stats['total_tps'] / success_count:.1f}")

        console.print()
        console.print(table)
        return self.results

    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Persist results as JSONL."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = RESULTS_DIR / f"{self.engine_label}_{self.profile_label}_{ts}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(result) + "\n")

        console.print(f"\n[bold green]✓ Results saved to:[/bold green] {output_path}")
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Mock Server (for CI / offline testing)
# ══════════════════════════════════════════════════════════════════════════════

async def _run_mock_server(port: int = 9999):
    """
    Minimal aiohttp mock server that echoes back fake streaming tokens.
    Used for validating the load generator pipeline without a real model.
    """
    from aiohttp import web

    async def completions_handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        n_tokens = body.get("max_tokens", 20)
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)

        for i in range(n_tokens):
            await asyncio.sleep(0.002)  # simulate 2ms inter-token latency
            chunk = {
                "id": "mock-completion",
                "object": "text_completion",
                "choices": [{"text": f" token_{i}", "index": 0, "finish_reason": None}],
            }
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    srv_app = web.Application()
    srv_app.router.add_post("/v1/completions", completions_handler)
    runner = web.AppRunner(srv_app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()
    console.print(f"[yellow]🔧 Mock server running on port {port}[/yellow]")
    return runner


# ══════════════════════════════════════════════════════════════════════════════
# Locust Integration
# ══════════════════════════════════════════════════════════════════════════════

try:
    from locust import HttpUser, between, task  # type: ignore

    class LLMUser(HttpUser):
        """
        Locust user class for distributed load testing.
        Run with: locust -f src/load_generator.py --headless -u 32 -r 4
        """
        wait_time = between(0.1, 0.5)
        host = os.getenv("VLLM_BASE_URL", "http://localhost:8000")

        PROMPTS = [
            "What is the capital of France?",
            "Explain the transformer architecture in 3 sentences.",
            "Write a Python function to reverse a linked list.",
        ]

        @task
        def completions_request(self):
            import random
            prompt = random.choice(self.PROMPTS)
            payload = {
                "model": os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.0,
            }
            with self.client.post("/v1/completions", json=payload, catch_response=True) as resp:
                if resp.status_code != 200:
                    resp.failure(f"Got status {resp.status_code}")

except ImportError:
    pass  # Locust not installed, that's fine


# ══════════════════════════════════════════════════════════════════════════════
# CLI Commands
# ══════════════════════════════════════════════════════════════════════════════

@app.command("run")
def run_load_test(
    engine: str = typer.Option("vllm", help="Engine to target: vllm | llamacpp | mock"),
    profile: str = typer.Option("short_qa", help="Load profile name from load_profiles.json"),
    concurrency: Optional[int] = typer.Option(None, help="Override default concurrency for the profile"),
    total_requests: Optional[int] = typer.Option(None, help="Override total request count"),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, help="Max tokens to generate per request"),
    base_url: Optional[str] = typer.Option(None, help="Override engine base URL"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[Path] = typer.Option(None, help="Output JSONL file path"),
):
    """Run a load test against the specified engine and traffic profile."""

    engine = engine.lower()
    if engine not in ENGINE_DEFAULTS and engine != "mock":
        console.print(f"[red]Unknown engine '{engine}'. Use: vllm | llamacpp | mock[/red]")
        raise typer.Exit(1)

    defaults = ENGINE_DEFAULTS.get(engine, ENGINE_DEFAULTS["mock"])
    _base_url = base_url or defaults["base_url"]
    _model = model or defaults["model"]

    # Load profile
    profiles_path = BENCHMARKS_DIR / "load_profiles.json"
    if not profiles_path.exists():
        console.print(f"[red]load_profiles.json not found at {profiles_path}[/red]")
        raise typer.Exit(1)

    with open(profiles_path) as f:
        all_profiles = json.load(f)["profiles"]

    if profile not in all_profiles:
        console.print(f"[red]Profile '{profile}' not found. Available: {list(all_profiles.keys())}[/red]")
        raise typer.Exit(1)

    prof = all_profiles[profile]
    _concurrency = concurrency or prof["default_concurrency"]
    _total_requests = total_requests or prof["total_requests"]
    prompts = prof["prompts"]

    console.print(Panel(
        f"[bold]Engine:[/bold] {engine} ({_base_url})\n"
        f"[bold]Model:[/bold]  {_model}\n"
        f"[bold]Profile:[/bold] {profile} — {prof['description'][:80]}\n"
        f"[bold]Requests:[/bold] {_total_requests} @ concurrency={_concurrency}",
        title="[bold cyan]🚀 LLM Load Generator",
        border_style="cyan",
    ))

    async def _main():
        runner = LoadTestRunner(_base_url, _model, engine, profile)

        if engine == "mock":
            mock_runner = await _run_mock_server(9999)
            await asyncio.sleep(0.2)  # let server start

        await runner.run(prompts, _total_requests, _concurrency, max_tokens)
        runner.save_results(output)

        if engine == "mock":
            await mock_runner.cleanup()

    asyncio.run(_main())


@app.command("mock")
def run_mock_test(
    total_requests: int = typer.Option(50, help="Number of mock requests"),
    concurrency: int = typer.Option(10, help="Concurrent requests"),
):
    """Quick smoke test using the built-in mock server. No GPU required."""

    async def _main():
        mock_runner = await _run_mock_server(9999)
        await asyncio.sleep(0.2)
        runner = LoadTestRunner("http://localhost:9999", "mock-model", "mock", "smoke_test")
        prompts = ["Test prompt for mock validation."] * 5
        await runner.run(prompts, total_requests, concurrency, max_tokens=10)
        runner.save_results()
        await mock_runner.cleanup()

    asyncio.run(_main())


if __name__ == "__main__":
    app()
