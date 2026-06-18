"""
metrics_collector.py — Hardware & Engine Telemetry Scraper
==========================================================
Scrapes Prometheus for system-level metrics during a benchmark run:
  • GPU VRAM usage (bytes used / total capacity)
  • GPU utilization % (via DCGM or nvidia-smi exporter)
  • CPU utilization %
  • vLLM-native metrics: cache hit rate, running sequences, queue depth

Also queries vLLM's /metrics endpoint directly for engine-level KPIs:
  • vllm:gpu_cache_usage_perc
  • vllm:num_requests_running
  • vllm:num_requests_waiting
  • vllm:e2e_request_latency_seconds_*

Can run in two modes:
  1. One-shot: snapshot current metrics
  2. Daemon: continuously poll at an interval, produce a time-series CSV

Usage:
  python src/metrics_collector.py snapshot --engine vllm
  python src/metrics_collector.py watch --engine vllm --interval 2 --duration 120
  python src/metrics_collector.py watch --engine llamacpp --output results/metrics.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))

console = Console()
app = typer.Typer(
    name="metrics-collector",
    help="Telemetry scraper for LLM benchmarking.",
    add_completion=False,
)


# ══════════════════════════════════════════════════════════════════════════════
# Prometheus Query Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _prometheus_query(metric: str, prom_url: str = PROMETHEUS_URL) -> Optional[float]:
    """Execute an instant Prometheus query. Returns scalar value or None."""
    try:
        import urllib.request
        import urllib.parse

        url = f"{prom_url}/api/v1/query?query={urllib.parse.quote(metric)}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        results = data.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
    except Exception:
        pass
    return None


def _vllm_direct_metrics(vllm_url: str = VLLM_BASE_URL) -> dict:
    """
    Scrape vLLM's native Prometheus /metrics endpoint directly.
    Returns a flat dict of metric_name → float value.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(f"{vllm_url}/metrics", timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return {}

    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            # Handle labels: metric{label="val"} value
            if "{" in line:
                name_part, value_str = line.rsplit("} ", 1)
                name = name_part.split("{")[0]
            else:
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                name, value_str = parts

            # Only capture known vLLM metrics
            if "vllm:" in name or "vllm_" in name:
                try:
                    metrics[name.strip()] = float(value_str.strip())
                except ValueError:
                    pass
        except Exception:
            continue
    return metrics


def _llamacpp_direct_metrics(llamacpp_url: str = LLAMACPP_BASE_URL) -> dict:
    """
    Scrape llama.cpp's /metrics endpoint (available in recent builds).
    Falls back to empty dict if endpoint not available.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(f"{llamacpp_url}/metrics", timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return {}

    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                metrics[parts[0].strip()] = float(parts[1].strip())
        except (ValueError, Exception):
            continue
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Metric Queries Per Engine
# ══════════════════════════════════════════════════════════════════════════════

# Prometheus metric expressions for infrastructure monitoring
# These work with nvidia-smi-exporter or DCGM exporter + node-exporter
PROMETHEUS_METRICS = {
    "gpu_vram_used_mb": "nvidia_smi_memory_used_bytes / 1024 / 1024",
    "gpu_vram_total_mb": "nvidia_smi_memory_total_bytes / 1024 / 1024",
    "gpu_utilization_pct": "nvidia_smi_utilization_gpu_ratio * 100",
    "gpu_temp_c": "nvidia_smi_temperature_gpu",
    "gpu_power_w": "nvidia_smi_power_draw_watts",
    "cpu_utilization_pct": '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[30s])) * 100)',
    "memory_used_gb": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1024^3",
}

# vLLM-native metrics (scraped from /metrics endpoint)
VLLM_METRICS = [
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]


def collect_snapshot(engine: str, prom_url: str = PROMETHEUS_URL) -> dict:
    """
    Collect a single snapshot of all metrics for the specified engine.
    Returns a flat dict ready for CSV/JSON serialization.
    """
    snapshot: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epoch_s": time.time(),
        "engine": engine,
    }

    # ── Prometheus infrastructure metrics ────────────────────────────────────
    for key, query in PROMETHEUS_METRICS.items():
        val = _prometheus_query(query, prom_url)
        snapshot[key] = round(val, 2) if val is not None else None

    # Derived: VRAM utilization %
    if snapshot.get("gpu_vram_used_mb") and snapshot.get("gpu_vram_total_mb"):
        snapshot["gpu_vram_utilization_pct"] = round(
            snapshot["gpu_vram_used_mb"] / snapshot["gpu_vram_total_mb"] * 100, 1
        )
    else:
        snapshot["gpu_vram_utilization_pct"] = None

    # ── Engine-native metrics ─────────────────────────────────────────────────
    if engine == "vllm":
        direct = _vllm_direct_metrics(VLLM_BASE_URL)
        for key in VLLM_METRICS:
            # Match by suffix in case of label variations
            matched = next((v for k, v in direct.items() if key in k), None)
            clean_key = key.replace("vllm:", "vllm_").replace(":", "_")
            snapshot[clean_key] = round(matched, 4) if matched is not None else None

    elif engine == "llamacpp":
        direct = _llamacpp_direct_metrics(LLAMACPP_BASE_URL)
        for k, v in direct.items():
            snapshot[f"llamacpp_{k}"] = v

    return snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Daemon Watcher
# ══════════════════════════════════════════════════════════════════════════════

class MetricsDaemon:
    """
    Background thread that continuously polls metrics at a fixed interval.
    Call start() to begin collection, stop() to halt and retrieve data.
    """

    def __init__(self, engine: str, interval_s: float = 2.0, prom_url: str = PROMETHEUS_URL):
        self.engine = engine
        self.interval_s = interval_s
        self.prom_url = prom_url
        self._snapshots: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start background metric collection thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        console.print(f"[dim]📡 Metrics daemon started (engine={self.engine}, interval={self.interval_s}s)[/dim]")

    def stop(self) -> list[dict]:
        """Stop collection and return all collected snapshots."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        console.print(f"[dim]📡 Metrics daemon stopped. Collected {len(self._snapshots)} snapshots.[/dim]")
        return self._snapshots

    def _collect_loop(self):
        while not self._stop_event.is_set():
            try:
                snapshot = collect_snapshot(self.engine, self.prom_url)
                self._snapshots.append(snapshot)
            except Exception as exc:
                self._snapshots.append({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "engine": self.engine,
                    "error": str(exc),
                })
            self._stop_event.wait(timeout=self.interval_s)

    @property
    def snapshots(self) -> list[dict]:
        return list(self._snapshots)


def save_metrics_csv(snapshots: list[dict], output_path: Path) -> Path:
    """Write collected snapshots to a CSV file."""
    if not snapshots:
        console.print("[yellow]No snapshots to save.[/yellow]")
        return output_path

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(snapshots[0].keys())
    # Ensure all snapshots have the same keys
    all_keys: set[str] = set()
    for s in snapshots:
        all_keys.update(s.keys())
    fieldnames = sorted(all_keys)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for snap in snapshots:
            writer.writerow({k: snap.get(k, "") for k in fieldnames})

    console.print(f"[bold green]✓ Metrics saved to:[/bold green] {output_path}")
    return output_path


def _make_live_table(snapshot: dict) -> Table:
    """Render a rich table for live display of current metrics."""
    table = Table(title="📡 Live Metrics Snapshot", show_header=True, header_style="bold blue", min_width=52)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    display_keys = [
        ("engine", "Engine"),
        ("timestamp_utc", "Timestamp"),
        ("gpu_vram_used_mb", "VRAM Used (MB)"),
        ("gpu_vram_total_mb", "VRAM Total (MB)"),
        ("gpu_vram_utilization_pct", "VRAM Utilization (%)"),
        ("gpu_utilization_pct", "GPU Compute (%)"),
        ("gpu_temp_c", "GPU Temp (°C)"),
        ("gpu_power_w", "GPU Power (W)"),
        ("cpu_utilization_pct", "CPU Utilization (%)"),
        ("memory_used_gb", "RAM Used (GB)"),
        ("vllm_gpu_cache_usage_perc", "vLLM KV Cache Usage (%)"),
        ("vllm_num_requests_running", "vLLM Requests Running"),
        ("vllm_num_requests_waiting", "vLLM Requests Queued"),
    ]

    for key, label in display_keys:
        val = snapshot.get(key)
        if val is not None:
            display_val = f"{val:.1f}" if isinstance(val, float) else str(val)
        else:
            display_val = "[dim]N/A[/dim]"
        table.add_row(label, display_val)

    return table


# ══════════════════════════════════════════════════════════════════════════════
# CLI Commands
# ══════════════════════════════════════════════════════════════════════════════

@app.command("snapshot")
def cmd_snapshot(
    engine: str = typer.Option("vllm", help="Engine to monitor: vllm | llamacpp"),
    prom_url: str = typer.Option(PROMETHEUS_URL, help="Prometheus base URL"),
):
    """Take a single metrics snapshot and display it."""
    console.print(f"[cyan]Taking metrics snapshot for engine=[bold]{engine}[/bold]...[/cyan]")
    snap = collect_snapshot(engine, prom_url)
    table = _make_live_table(snap)
    console.print(table)
    console.print()
    console.print("[dim]Raw JSON:[/dim]")
    console.print_json(json.dumps(snap, default=str))


@app.command("watch")
def cmd_watch(
    engine: str = typer.Option("vllm", help="Engine to monitor: vllm | llamacpp"),
    interval: float = typer.Option(2.0, help="Poll interval in seconds"),
    duration: Optional[int] = typer.Option(None, help="Duration to watch in seconds (None = until Ctrl+C)"),
    output: Optional[Path] = typer.Option(None, help="Output CSV path"),
    prom_url: str = typer.Option(PROMETHEUS_URL, help="Prometheus base URL"),
):
    """
    Continuously watch and display metrics. Press Ctrl+C to stop.
    Saves collected data to CSV when finished.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output or RESULTS_DIR / f"metrics_{engine}_{ts}.csv"

    daemon = MetricsDaemon(engine, interval, prom_url)
    daemon.start()

    start = time.time()
    try:
        with Live(console=console, refresh_per_second=1 / interval) as live:
            while True:
                if duration and (time.time() - start) >= duration:
                    break
                snap = collect_snapshot(engine, prom_url)
                live.update(_make_live_table(snap))
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping metrics collection...[/yellow]")

    snapshots = daemon.stop()
    save_metrics_csv(snapshots, output_path)


if __name__ == "__main__":
    app()
