"""
cost_calculator.py — Financial Cost Modeller
=============================================
Reads a benchmark results JSONL file and computes the cost of running
the workload against a configurable cloud GPU pricing model.

Outputs:
  • cost_per_1k_requests  (USD)
  • throughput_rps         (requests per second)
  • p50/p95/p99 latency    (milliseconds)
  • gpu_efficiency_score   (throughput relative to cost)
  • Full JSON/CSV summary per configuration

Cost Model:
  Cloud GPU pricing is passed as $/GPU-hour. The calculator determines
  how many GPU-hours are consumed per 1,000 requests based on observed
  throughput, then multiplies by the hourly rate.

  Formula:
    seconds_per_1k = 1000 / throughput_rps
    gpu_hours_per_1k = seconds_per_1k / 3600
    cost_per_1k = gpu_hours_per_1k * gpu_cost_per_hour

Usage:
  python src/cost_calculator.py compute --results results/vllm_short_qa_20240101.jsonl
  python src/cost_calculator.py compare --results-dir results/
  python src/cost_calculator.py compare --results-dir results/ --format table
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
GPU_COST_PER_HOUR = float(os.getenv("GPU_COST_PER_HOUR", "3.00"))
GPU_INSTANCE_LABEL = os.getenv("GPU_INSTANCE_LABEL", "A100-40GB")

# Reference cloud GPU pricing table (for display)
CLOUD_GPU_PRICING = {
    "A100-40GB":  {"provider": "Lambda Labs",   "cost_per_hour": 3.00,  "vram_gb": 40},
    "A100-80GB":  {"provider": "Lambda Labs",   "cost_per_hour": 4.00,  "vram_gb": 80},
    "H100-80GB":  {"provider": "Lambda Labs",   "cost_per_hour": 8.00,  "vram_gb": 80},
    "RTX-4090":   {"provider": "Vast.ai",       "cost_per_hour": 1.20,  "vram_gb": 24},
    "RTX-3090":   {"provider": "Vast.ai",       "cost_per_hour": 0.80,  "vram_gb": 24},
    "A10G":       {"provider": "AWS g5.xlarge", "cost_per_hour": 1.006, "vram_gb": 24},
    "V100-16GB":  {"provider": "AWS p3.2xl",    "cost_per_hour": 3.06,  "vram_gb": 16},
    "T4":         {"provider": "GCP n1+T4",     "cost_per_hour": 0.35,  "vram_gb": 16},
}

console = Console()
app = typer.Typer(
    name="cost-calculator",
    help="Financial cost modeller for LLM inference benchmarks.",
    add_completion=False,
)


# ══════════════════════════════════════════════════════════════════════════════
# Core Calculation Logic
# ══════════════════════════════════════════════════════════════════════════════

def load_results_jsonl(path: Path) -> list[dict]:
    """Load benchmark JSONL results file."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_latency_percentiles(latencies_ms: list[float]) -> dict:
    """Compute p50, p95, p99 from a list of latency values."""
    if not latencies_ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "min_ms": None, "max_ms": None}
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    return {
        "p50_ms": round(sorted_l[int(n * 0.50)], 2),
        "p95_ms": round(sorted_l[int(n * 0.95)], 2),
        "p99_ms": round(sorted_l[min(int(n * 0.99), n - 1)], 2),
        "min_ms": round(sorted_l[0], 2),
        "max_ms": round(sorted_l[-1], 2),
        "mean_ms": round(sum(sorted_l) / n, 2),
    }


def calculate_cost(
    results: list[dict],
    gpu_cost_per_hour: float = GPU_COST_PER_HOUR,
    gpu_instance_label: str = GPU_INSTANCE_LABEL,
) -> dict:
    """
    Compute cost metrics from a list of benchmark result dicts.

    Returns a comprehensive cost report dict.
    """
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    if not successful:
        return {"error": "No successful requests in results file."}

    # ── Latency percentiles ──────────────────────────────────────────────────
    latencies_ms = [r["total_latency_ms"] for r in successful if r.get("total_latency_ms")]
    ttfts_ms = [r["ttft_ms"] for r in successful if r.get("ttft_ms")]
    latency_stats = compute_latency_percentiles(latencies_ms)
    ttft_stats = compute_latency_percentiles(ttfts_ms)

    # ── Throughput ────────────────────────────────────────────────────────────
    # Use wall-clock total span of the run (from first to last timestamp)
    timestamps = []
    for r in results:
        try:
            timestamps.append(datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00")))
        except Exception:
            pass

    if len(timestamps) >= 2:
        span_s = (max(timestamps) - min(timestamps)).total_seconds()
    else:
        # Fallback: sum of all latencies / concurrency estimate
        span_s = sum(latencies_ms) / 1000 / max(1, len(successful))

    span_s = max(span_s, 0.001)  # avoid division by zero
    throughput_rps = len(results) / span_s

    # ── Token stats ───────────────────────────────────────────────────────────
    token_counts = [r.get("tokens_generated", 0) for r in successful]
    total_tokens = sum(token_counts)
    avg_tps_per_req = sum(r.get("tokens_per_second", 0) for r in successful) / len(successful)

    # ── Cost model ────────────────────────────────────────────────────────────
    # How many seconds does 1,000 requests take at observed throughput?
    seconds_per_1k_requests = 1000.0 / throughput_rps
    gpu_hours_per_1k_requests = seconds_per_1k_requests / 3600.0
    cost_per_1k_requests = gpu_hours_per_1k_requests * gpu_cost_per_hour

    # Cost per 1M tokens generated
    tokens_per_second_aggregate = total_tokens / span_s
    cost_per_1m_tokens = (1_000_000 / tokens_per_second_aggregate / 3600) * gpu_cost_per_hour if tokens_per_second_aggregate > 0 else None

    # GPU efficiency score: throughput per dollar (higher = better)
    gpu_efficiency = throughput_rps / gpu_cost_per_hour

    # ── Assemble report ───────────────────────────────────────────────────────
    meta = {
        "engine": results[0].get("run_id", "unknown") if results else "unknown",
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "error_rate_pct": round(len(failed) / len(results) * 100, 2),
        "wall_time_s": round(span_s, 2),
    }

    cost_report = {
        **meta,
        # Throughput
        "throughput_rps": round(throughput_rps, 3),
        "requests_per_minute": round(throughput_rps * 60, 1),
        "requests_per_hour": round(throughput_rps * 3600, 0),
        # Latency
        **{f"total_latency_{k}": v for k, v in latency_stats.items()},
        **{f"ttft_{k}": v for k, v in ttft_stats.items()},
        # Token throughput
        "avg_tokens_generated_per_request": round(total_tokens / len(successful), 1) if successful else 0,
        "aggregate_tokens_per_second": round(tokens_per_second_aggregate, 2),
        "avg_tokens_per_second_per_request": round(avg_tps_per_req, 2),
        # Cost
        "gpu_instance": gpu_instance_label,
        "gpu_cost_per_hour_usd": gpu_cost_per_hour,
        "cost_per_1k_requests_usd": round(cost_per_1k_requests, 6),
        "cost_per_1m_tokens_usd": round(cost_per_1m_tokens, 4) if cost_per_1m_tokens else None,
        "gpu_efficiency_score": round(gpu_efficiency, 3),
        "gpu_hours_per_1k_requests": round(gpu_hours_per_1k_requests, 6),
        # Meta
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
    return cost_report


def print_cost_report(report: dict, title: str = "Cost Analysis Report"):
    """Render a rich table from a cost report dict."""
    table = Table(title=f"💰 {title}", show_header=True, header_style="bold green", min_width=60)
    table.add_column("Metric", style="cyan", min_width=36)
    table.add_column("Value", style="white", justify="right")

    sections = [
        ("── Workload ──────────────────────────────────", None),
        ("Total Requests", report.get("total_requests")),
        ("Successful", report.get("successful_requests")),
        ("Failed", report.get("failed_requests")),
        ("Error Rate (%)", report.get("error_rate_pct")),
        ("Wall Time (s)", report.get("wall_time_s")),
        ("── Throughput ────────────────────────────────", None),
        ("Throughput (RPS)", report.get("throughput_rps")),
        ("Requests / Hour", f"{report.get('requests_per_hour', 0):,.0f}"),
        ("── Latency (Total Response) ─────────────────", None),
        ("P50 Latency (ms)", report.get("total_latency_p50_ms")),
        ("P95 Latency (ms)", report.get("total_latency_p95_ms")),
        ("P99 Latency (ms)", report.get("total_latency_p99_ms")),
        ("Mean Latency (ms)", report.get("total_latency_mean_ms")),
        ("── TTFT (Time to First Token) ───────────────", None),
        ("P50 TTFT (ms)", report.get("ttft_p50_ms")),
        ("P95 TTFT (ms)", report.get("ttft_p95_ms")),
        ("── Token Throughput ──────────────────────────", None),
        ("Avg Tokens / Request", report.get("avg_tokens_generated_per_request")),
        ("Aggregate Tokens/Sec", report.get("aggregate_tokens_per_second")),
        ("── Cost Model ───────────────────────────────", None),
        ("GPU Instance", report.get("gpu_instance")),
        ("GPU Cost ($/hr)", f"${report.get('gpu_cost_per_hour_usd', 0):.2f}"),
        ("Cost per 1K Requests ($)", f"${report.get('cost_per_1k_requests_usd', 0):.6f}"),
        ("Cost per 1M Tokens ($)", f"${report.get('cost_per_1m_tokens_usd', 0):.4f}" if report.get("cost_per_1m_tokens_usd") else "N/A"),
        ("GPU Efficiency Score", f"{report.get('gpu_efficiency_score', 0):.3f} RPS/$"),
    ]

    for label, value in sections:
        if value is None and label.startswith("──"):
            table.add_section()
            table.add_row(f"[bold dim]{label}[/bold dim]", "")
        elif value is not None:
            table.add_row(label, str(value))

    console.print()
    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Commands
# ══════════════════════════════════════════════════════════════════════════════

@app.command("compute")
def cmd_compute(
    results: Path = typer.Argument(..., help="Path to a results JSONL file"),
    gpu_cost: float = typer.Option(GPU_COST_PER_HOUR, help="GPU cost in USD per hour"),
    gpu_label: str = typer.Option(GPU_INSTANCE_LABEL, help="GPU instance label for display"),
    output: Optional[Path] = typer.Option(None, help="Save JSON report to this path"),
    show_pricing: bool = typer.Option(False, help="Show reference GPU pricing table"),
):
    """Compute cost metrics for a single benchmark results file."""
    if show_pricing:
        pt = Table(title="Reference Cloud GPU Pricing", header_style="bold yellow")
        pt.add_column("Instance")
        pt.add_column("Provider")
        pt.add_column("VRAM (GB)", justify="right")
        pt.add_column("$/hr", justify="right")
        for label, info in CLOUD_GPU_PRICING.items():
            pt.add_row(label, info["provider"], str(info["vram_gb"]), f"${info['cost_per_hour']:.3f}")
        console.print(pt)
        console.print()

    if not results.exists():
        console.print(f"[red]Results file not found: {results}[/red]")
        raise typer.Exit(1)

    data = load_results_jsonl(results)
    report = calculate_cost(data, gpu_cost, gpu_label)

    if "error" in report:
        console.print(f"[red]{report['error']}[/red]")
        raise typer.Exit(1)

    print_cost_report(report, title=f"Cost Analysis — {results.name}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"[bold green]✓ Report saved to:[/bold green] {output}")


@app.command("compare")
def cmd_compare(
    results_dir: Path = typer.Option(RESULTS_DIR, help="Directory containing JSONL results files"),
    gpu_cost: float = typer.Option(GPU_COST_PER_HOUR, help="GPU cost in USD per hour"),
    gpu_label: str = typer.Option(GPU_INSTANCE_LABEL, help="GPU instance label"),
    sort_by: str = typer.Option("cost_per_1k_requests_usd", help="Column to sort results by"),
):
    """Compare cost metrics across all results files in a directory."""
    jsonl_files = list(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        console.print(f"[yellow]No .jsonl files found in {results_dir}[/yellow]")
        raise typer.Exit(0)

    reports = []
    for f in jsonl_files:
        try:
            data = load_results_jsonl(f)
            report = calculate_cost(data, gpu_cost, gpu_label)
            report["source_file"] = f.name
            reports.append(report)
        except Exception as e:
            console.print(f"[yellow]Skipping {f.name}: {e}[/yellow]")

    if not reports:
        console.print("[red]No valid reports generated.[/red]")
        raise typer.Exit(1)

    # Sort
    try:
        reports.sort(key=lambda r: r.get(sort_by) or float("inf"))
    except Exception:
        pass

    # Comparison table
    table = Table(title=f"📊 Benchmark Cost Comparison ({gpu_label} @ ${gpu_cost}/hr)", header_style="bold magenta")
    cols = [
        ("Source File", "source_file"),
        ("RPS", "throughput_rps"),
        ("P50 (ms)", "total_latency_p50_ms"),
        ("P95 (ms)", "total_latency_p95_ms"),
        ("Avg TTFT (ms)", "ttft_mean_ms"),
        ("$/1K Req", "cost_per_1k_requests_usd"),
        ("$/1M Tok", "cost_per_1m_tokens_usd"),
        ("Eff. Score", "gpu_efficiency_score"),
        ("Errors %", "error_rate_pct"),
    ]
    for label, _ in cols:
        table.add_column(label, justify="right" if label != "Source File" else "left")

    for r in reports:
        table.add_row(
            r.get("source_file", "?"),
            str(r.get("throughput_rps", "?")),
            str(r.get("total_latency_p50_ms", "?")),
            str(r.get("total_latency_p95_ms", "?")),
            str(r.get("ttft_mean_ms", "?")),
            f"${r.get('cost_per_1k_requests_usd', 0):.6f}",
            f"${r.get('cost_per_1m_tokens_usd', 0):.4f}" if r.get("cost_per_1m_tokens_usd") else "N/A",
            str(r.get("gpu_efficiency_score", "?")),
            f"{r.get('error_rate_pct', 0):.1f}%",
        )

    console.print()
    console.print(table)

    # Save combined CSV
    output_path = results_dir / "cost_comparison.json"
    with open(output_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    console.print(f"\n[bold green]✓ Comparison saved to:[/bold green] {output_path}")


if __name__ == "__main__":
    app()
