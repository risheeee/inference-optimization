"""
benchmark_runner.py — Master Benchmark Matrix Orchestrator
==========================================================
Automates the full benchmark sweep across all combinations of:

  • Engine:         vllm | llamacpp
  • Weight Format:  fp16 (baseline) | awq_4bit | gptq_8bit | gguf_q4 | gguf_q8
  • KV Cache:       fp16 (auto) | fp8
  • Batch Strategy: static_1 | static_8 | static_32 | continuous
  • Load Profile:   short_qa | medium_reasoning | long_context | burst_spike

For each combination, the runner:
  1. Patches the engine's config.yaml with the correct parameters
  2. Restarts the Docker Compose service and waits for health
  3. Runs the load generator (src/load_generator.py)
  4. Runs the quality evaluator (src/quality_evaluator.py)
  5. Runs the cost calculator (src/cost_calculator.py)
  6. Aggregates all results into results/benchmark_matrix.csv

Usage:
  # Full automated sweep (takes hours)
  python src/benchmark_runner.py sweep

  # Run a specific subset of combinations
  python src/benchmark_runner.py sweep --engines vllm --profiles short_qa medium_reasoning

  # Dry run to see what would execute
  python src/benchmark_runner.py sweep --dry-run

  # Run a single combination manually
  python src/benchmark_runner.py run-single \\
    --engine vllm --weight-format awq_4bit \\
    --kv-cache fp8 --batch-strategy continuous \\
    --profile short_qa
"""

from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
ENGINES_DIR = ROOT_DIR / "engines"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
SRC_DIR = ROOT_DIR / "src"

console = Console()
app = typer.Typer(
    name="benchmark-runner",
    help="Master orchestrator for the LLM benchmark matrix sweep.",
    add_completion=False,
)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Matrix Definition
# ══════════════════════════════════════════════════════════════════════════════

# Each entry defines a named configuration and how it patches config.yaml
WEIGHT_FORMAT_CONFIGS = {
    "fp16_baseline": {
        "vllm": {"dtype": "auto", "quantization": None, "kv_cache_dtype": "auto"},
        "llamacpp": {"model_path": "/models/Meta-Llama-3-8B-Instruct-F16.gguf"},
        "label": "FP16 Baseline",
    },
    "awq_4bit": {
        "vllm": {"dtype": "auto", "quantization": "awq", "kv_cache_dtype": "auto"},
        "llamacpp": None,  # AWQ not natively supported in llama.cpp
        "label": "AWQ 4-bit",
    },
    "gptq_8bit": {
        "vllm": {"dtype": "auto", "quantization": "gptq", "kv_cache_dtype": "auto"},
        "llamacpp": None,
        "label": "GPTQ 8-bit",
    },
    "gguf_q4": {
        "vllm": None,  # GGUF is native to llama.cpp
        "llamacpp": {"model_path": "/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"},
        "label": "GGUF Q4_K_M (4-bit)",
    },
    "gguf_q8": {
        "vllm": None,
        "llamacpp": {"model_path": "/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf"},
        "label": "GGUF Q8_0 (8-bit)",
    },
}

KV_CACHE_CONFIGS = {
    "fp16_kvcache": {
        "vllm": {"kv_cache_dtype": "auto"},
        "llamacpp": {},  # llama.cpp uses native precision
        "label": "KV Cache FP16",
    },
    "fp8_kvcache": {
        "vllm": {"kv_cache_dtype": "fp8"},
        "llamacpp": {},  # FP8 KV not applicable to llama.cpp
        "label": "KV Cache FP8",
    },
}

BATCHING_CONFIGS = {
    "static_b1": {
        "vllm": {"max_num_seqs": 1, "enable_chunked_prefill": False},
        "llamacpp": {"n_batch": 1, "n_parallel": 1, "cont_batching": False},
        "label": "Static Batch=1",
    },
    "static_b8": {
        "vllm": {"max_num_seqs": 8, "enable_chunked_prefill": False},
        "llamacpp": {"n_batch": 8, "n_parallel": 2, "cont_batching": False},
        "label": "Static Batch=8",
    },
    "static_b32": {
        "vllm": {"max_num_seqs": 32, "enable_chunked_prefill": False},
        "llamacpp": {"n_batch": 512, "n_parallel": 8, "cont_batching": False},
        "label": "Static Batch=32",
    },
    "continuous": {
        "vllm": {"max_num_seqs": 256, "enable_chunked_prefill": True},
        "llamacpp": {"n_batch": 512, "n_parallel": 8, "cont_batching": True},
        "label": "Continuous/In-Flight Batching",
    },
}

LOAD_PROFILES = ["short_qa", "medium_reasoning", "long_context", "burst_spike"]

# ── Default sweep (a representative subset, not exhaustive) ──────────────────
DEFAULT_SWEEP = [
    # (engine, weight_format, kv_cache, batch_strategy)
    ("vllm",     "fp16_baseline", "fp16_kvcache", "continuous"),     # Baseline
    ("vllm",     "awq_4bit",      "fp16_kvcache", "continuous"),     # Weight quant
    ("vllm",     "awq_4bit",      "fp8_kvcache",  "continuous"),     # Weight + KV quant
    ("vllm",     "fp16_baseline", "fp16_kvcache", "static_b32"),     # Static batching
    ("vllm",     "fp16_baseline", "fp16_kvcache", "static_b1"),      # Single-request baseline
    ("llamacpp", "fp16_baseline", "fp16_kvcache", "static_b32"),     # llama.cpp FP16
    ("llamacpp", "gguf_q4",       "fp16_kvcache", "static_b32"),     # GGUF 4-bit
    ("llamacpp", "gguf_q8",       "fp16_kvcache", "static_b32"),     # GGUF 8-bit
    ("llamacpp", "gguf_q4",       "fp16_kvcache", "continuous"),     # GGUF + cont. batching
]


# ══════════════════════════════════════════════════════════════════════════════
# Config Patching
# ══════════════════════════════════════════════════════════════════════════════

def load_config(engine: str) -> dict:
    """Load the engine's config.yaml."""
    config_path = ENGINES_DIR / engine / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(engine: str, config: dict):
    """Write a config dict back to the engine's config.yaml."""
    # Normalize engine dir name
    engine_dir = "vllm" if engine == "vllm" else "llamacpp"
    config_path = ENGINES_DIR / engine_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def patch_config(engine: str, weight_fmt: str, kv_cache: str, batch_strategy: str) -> dict:
    """
    Build a patched config dict for the given combination.
    Returns the patched config dict (does NOT write to disk yet).
    """
    base = load_config("vllm" if engine == "vllm" else "llamacpp")
    patched = copy.deepcopy(base)

    def apply(patch_dict: Optional[dict]):
        if patch_dict:
            patched.update(patch_dict)

    engine_key = engine  # "vllm" or "llamacpp"

    # Apply weight format patches
    wf_patch = WEIGHT_FORMAT_CONFIGS.get(weight_fmt, {}).get(engine_key)
    apply(wf_patch)

    # Apply KV cache patches
    kv_patch = KV_CACHE_CONFIGS.get(kv_cache, {}).get(engine_key)
    apply(kv_patch)

    # Apply batching patches
    bs_patch = BATCHING_CONFIGS.get(batch_strategy, {}).get(engine_key)
    apply(bs_patch)

    return patched


# ══════════════════════════════════════════════════════════════════════════════
# Docker Compose Service Management
# ══════════════════════════════════════════════════════════════════════════════

def _run_subprocess(cmd: list[str], cwd: Path = ROOT_DIR, capture: bool = True) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined output)."""
    result = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True, timeout=300,
    )
    output = result.stdout or ""
    return result.returncode, output


def restart_engine_service(engine: str, dry_run: bool = False) -> bool:
    """
    Restart the engine's Docker Compose service.
    Returns True if successful.
    """
    service_name = f"{engine}-engine"
    profile = engine  # docker-compose profile matches engine name

    console.print(f"[dim]🐳 Restarting Docker service '{service_name}'...[/dim]")
    if dry_run:
        console.print(f"[dim]  [DRY RUN] docker compose --profile {profile} restart {service_name}[/dim]")
        return True

    # Restart service
    rc, out = _run_subprocess(["docker", "compose", "--profile", profile, "restart", service_name])
    if rc != 0:
        console.print(f"[red]Failed to restart {service_name}:\n{out}[/red]")
        return False

    return wait_for_engine_health(engine)


def wait_for_engine_health(engine: str, timeout_s: int = 120, poll_interval: float = 3.0) -> bool:
    """Poll the engine's health endpoint until it responds OK or times out."""
    import urllib.request
    import urllib.error

    ENGINE_HEALTH = {
        "vllm": f"{os.getenv('VLLM_BASE_URL', 'http://localhost:8000')}/health",
        "llamacpp": f"{os.getenv('LLAMACPP_BASE_URL', 'http://localhost:8080')}/health",
    }
    url = ENGINE_HEALTH.get(engine, "http://localhost:8000/health")
    deadline = time.time() + timeout_s

    console.print(f"[dim]⏳ Waiting for {engine} to be healthy ({url})...[/dim]")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    console.print(f"[green]✓ {engine} is healthy[/green]")
                    return True
        except Exception:
            pass
        time.sleep(poll_interval)

    console.print(f"[red]✗ {engine} did not become healthy within {timeout_s}s[/red]")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Single Run Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_single_combination(
    engine: str,
    weight_format: str,
    kv_cache: str,
    batch_strategy: str,
    profile: str,
    dry_run: bool = False,
    skip_quality: bool = False,
    restart_docker: bool = True,
) -> dict:
    """
    Execute a single benchmark combination end-to-end.
    Returns a result summary dict.
    """
    config_label = f"{engine}__{weight_format}__{kv_cache}__{batch_strategy}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(Panel(
        f"[bold]Engine:[/bold]         {engine}\n"
        f"[bold]Weight Format:[/bold]  {WEIGHT_FORMAT_CONFIGS.get(weight_format, {}).get('label', weight_format)}\n"
        f"[bold]KV Cache:[/bold]       {KV_CACHE_CONFIGS.get(kv_cache, {}).get('label', kv_cache)}\n"
        f"[bold]Batch Strategy:[/bold] {BATCHING_CONFIGS.get(batch_strategy, {}).get('label', batch_strategy)}\n"
        f"[bold]Load Profile:[/bold]   {profile}",
        title=f"[bold cyan]⚙ Combination: {config_label}",
        border_style="cyan",
    ))

    result_row = {
        "engine": engine,
        "weight_format": weight_format,
        "weight_format_label": WEIGHT_FORMAT_CONFIGS.get(weight_format, {}).get("label", weight_format),
        "kv_cache": kv_cache,
        "kv_cache_label": KV_CACHE_CONFIGS.get(kv_cache, {}).get("label", kv_cache),
        "batch_strategy": batch_strategy,
        "batch_label": BATCHING_CONFIGS.get(batch_strategy, {}).get("label", batch_strategy),
        "load_profile": profile,
        "config_label": config_label,
        "timestamp": ts,
        "status": "pending",
    }

    # ── 1. Patch and write engine config ─────────────────────────────────────
    try:
        patched = patch_config(engine, weight_format, kv_cache, batch_strategy)
        if not dry_run:
            save_config(engine, patched)
            console.print(f"[dim]✓ Config patched for {engine}[/dim]")
    except Exception as e:
        console.print(f"[red]Config patch failed: {e}[/red]")
        result_row["status"] = "config_error"
        result_row["error"] = str(e)
        return result_row

    # ── 2. Restart Docker service ─────────────────────────────────────────────
    if restart_docker:
        healthy = restart_engine_service(engine, dry_run)
        if not healthy and not dry_run:
            result_row["status"] = "engine_unhealthy"
            result_row["error"] = "Engine did not become healthy after restart"
            return result_row

    # ── 3. Run load generator ─────────────────────────────────────────────────
    load_results_path = RESULTS_DIR / f"{config_label}_{profile}_{ts}.jsonl"
    load_cmd = [
        sys.executable, str(SRC_DIR / "load_generator.py"),
        "run",
        "--engine", engine,
        "--profile", profile,
        "--output", str(load_results_path),
    ]
    console.print(f"\n[cyan]▶ Running load generator...[/cyan]")
    if dry_run:
        console.print(f"[dim]  [DRY RUN] {' '.join(load_cmd)}[/dim]")
    else:
        rc, out = _run_subprocess(load_cmd, capture=False)
        if rc != 0:
            result_row["status"] = "load_generator_error"
            result_row["error"] = f"Load generator exited with code {rc}"
            return result_row

    # ── 4. Run cost calculator ────────────────────────────────────────────────
    cost_report_path = RESULTS_DIR / f"cost_{config_label}_{profile}_{ts}.json"
    cost_cmd = [
        sys.executable, str(SRC_DIR / "cost_calculator.py"),
        "compute",
        str(load_results_path),
        "--output", str(cost_report_path),
    ]
    console.print(f"\n[cyan]▶ Computing costs...[/cyan]")
    if dry_run:
        console.print(f"[dim]  [DRY RUN] {' '.join(cost_cmd)}[/dim]")
        cost_data = {}
    else:
        rc, _ = _run_subprocess(cost_cmd, capture=False)
        cost_data = {}
        if cost_report_path.exists():
            with open(cost_report_path) as f:
                cost_data = json.load(f)

    # ── 5. Run quality evaluator ──────────────────────────────────────────────
    quality_data = {}
    if not skip_quality:
        quality_report_path = RESULTS_DIR / f"quality_{config_label}_{ts}.json"
        quality_cmd = [
            sys.executable, str(SRC_DIR / "quality_evaluator.py"),
            "evaluate",
            "--engine", engine,
            "--config-label", config_label,
            "--max-items", "20",  # Quick quality check per combination (20 items)
            "--output", str(quality_report_path),
        ]
        console.print(f"\n[cyan]▶ Evaluating output quality...[/cyan]")
        if dry_run:
            console.print(f"[dim]  [DRY RUN] {' '.join(quality_cmd)}[/dim]")
        else:
            rc, _ = _run_subprocess(quality_cmd, capture=False)
            if quality_report_path.exists():
                with open(quality_report_path) as f:
                    quality_data = json.load(f)

    # ── 6. Aggregate results ──────────────────────────────────────────────────
    result_row.update({
        "status": "success" if not dry_run else "dry_run",
        # Cost metrics
        "throughput_rps":             cost_data.get("throughput_rps"),
        "p50_latency_ms":             cost_data.get("total_latency_p50_ms"),
        "p95_latency_ms":             cost_data.get("total_latency_p95_ms"),
        "p99_latency_ms":             cost_data.get("total_latency_p99_ms"),
        "mean_ttft_ms":               cost_data.get("ttft_mean_ms"),
        "avg_tokens_per_second":      cost_data.get("aggregate_tokens_per_second"),
        "cost_per_1k_requests_usd":   cost_data.get("cost_per_1k_requests_usd"),
        "cost_per_1m_tokens_usd":     cost_data.get("cost_per_1m_tokens_usd"),
        "gpu_efficiency_score":       cost_data.get("gpu_efficiency_score"),
        "error_rate_pct":             cost_data.get("error_rate_pct"),
        # Quality metrics
        "json_accuracy_pct":          quality_data.get("json_extraction", {}).get("accuracy_pct"),
        "reasoning_score_avg":        quality_data.get("reasoning", {}).get("avg_score_1_to_5"),
        "composite_quality_score":    quality_data.get("composite_quality_score_0_to_1"),
        # Paths
        "load_results_file":          str(load_results_path),
        "cost_report_file":           str(cost_report_path) if not skip_quality else None,
        "quality_report_file":        str(quality_report_path) if not skip_quality else None,
    })
    return result_row


# ══════════════════════════════════════════════════════════════════════════════
# CLI Commands
# ══════════════════════════════════════════════════════════════════════════════

@app.command("sweep")
def cmd_sweep(
    engines: Optional[list[str]] = typer.Option(None, help="Engines to include: vllm llamacpp"),
    profiles: Optional[list[str]] = typer.Option(None, help="Load profiles to include"),
    dry_run: bool = typer.Option(False, help="Print planned combinations without running"),
    skip_quality: bool = typer.Option(False, help="Skip quality evaluation (faster)"),
    no_docker: bool = typer.Option(False, help="Skip Docker restart (use currently running engine)"),
    output_csv: Optional[Path] = typer.Option(None, help="Output CSV for benchmark matrix results"),
):
    """
    Run the full benchmark matrix sweep.
    Iterates over DEFAULT_SWEEP combinations; filter with --engines / --profiles.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter combinations
    combinations = [
        c for c in DEFAULT_SWEEP
        if (not engines or c[0] in engines)
    ]

    # Apply profile filter or use all profiles
    _profiles = profiles or LOAD_PROFILES[:2]  # default: short_qa + medium_reasoning

    console.print(Panel(
        f"[bold]Combinations:[/bold] {len(combinations)}\n"
        f"[bold]Profiles:[/bold]    {', '.join(_profiles)}\n"
        f"[bold]Total Runs:[/bold]  {len(combinations) * len(_profiles)}\n"
        f"[bold]Dry Run:[/bold]     {dry_run}",
        title="[bold magenta]🔬 Benchmark Matrix Sweep",
        border_style="magenta",
    ))

    if dry_run:
        # Print all planned runs
        table = Table(title="Planned Combinations (Dry Run)", header_style="bold yellow")
        for col in ["Engine", "Weight Format", "KV Cache", "Batch Strategy", "Profile"]:
            table.add_column(col)
        for (eng, wf, kv, bs) in combinations:
            for prof in _profiles:
                table.add_row(eng, wf, kv, bs, prof)
        console.print(table)
        return

    # Execute combinations
    all_results = []
    ts_sweep = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_csv or RESULTS_DIR / f"benchmark_matrix_{ts_sweep}.csv"

    total = len(combinations) * len(_profiles)
    run_num = 0

    for (engine, weight_fmt, kv_cache, batch_strat) in combinations:
        # Skip invalid combinations (e.g., AWQ on llama.cpp)
        wf_cfg = WEIGHT_FORMAT_CONFIGS.get(weight_fmt, {})
        if wf_cfg.get(engine) is None and weight_fmt != "fp16_baseline":
            console.print(f"[yellow]⏭ Skipping {engine}+{weight_fmt} (incompatible combination)[/yellow]")
            continue

        for profile in _profiles:
            run_num += 1
            console.print(f"\n[bold]─── Run {run_num}/{total} ───────────────────────────────[/bold]")
            result = run_single_combination(
                engine=engine,
                weight_format=weight_fmt,
                kv_cache=kv_cache,
                batch_strategy=batch_strat,
                profile=profile,
                dry_run=dry_run,
                skip_quality=skip_quality,
                restart_docker=not no_docker,
            )
            all_results.append(result)

            # Incremental CSV write (survives interruption)
            _write_csv(all_results, csv_path)
            console.print(f"[dim]✓ Results appended to {csv_path}[/dim]")

    # ── Final summary table ───────────────────────────────────────────────────
    console.print("\n")
    _print_matrix_summary(all_results)
    console.print(f"\n[bold green]✓ Full benchmark matrix saved to:[/bold green] {csv_path}")


@app.command("run-single")
def cmd_run_single(
    engine: str = typer.Option("vllm", help="Engine: vllm | llamacpp"),
    weight_format: str = typer.Option("fp16_baseline", help="Weight format key"),
    kv_cache: str = typer.Option("fp16_kvcache", help="KV cache config key"),
    batch_strategy: str = typer.Option("continuous", help="Batch strategy key"),
    profile: str = typer.Option("short_qa", help="Load profile"),
    dry_run: bool = typer.Option(False),
    skip_quality: bool = typer.Option(False),
    no_docker: bool = typer.Option(False),
):
    """Run a single benchmark combination."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_single_combination(
        engine, weight_format, kv_cache, batch_strategy, profile,
        dry_run=dry_run, skip_quality=skip_quality, restart_docker=not no_docker,
    )
    console.print_json(json.dumps(result, indent=2, default=str))


@app.command("list-configs")
def cmd_list_configs():
    """Print all available benchmark matrix configuration keys."""
    console.print("\n[bold cyan]Weight Formats:[/bold cyan]")
    for k, v in WEIGHT_FORMAT_CONFIGS.items():
        console.print(f"  [green]{k}[/green] — {v['label']}")
    console.print("\n[bold cyan]KV Cache Configs:[/bold cyan]")
    for k, v in KV_CACHE_CONFIGS.items():
        console.print(f"  [green]{k}[/green] — {v['label']}")
    console.print("\n[bold cyan]Batch Strategies:[/bold cyan]")
    for k, v in BATCHING_CONFIGS.items():
        console.print(f"  [green]{k}[/green] — {v['label']}")
    console.print("\n[bold cyan]Load Profiles:[/bold cyan]")
    for p in LOAD_PROFILES:
        console.print(f"  [green]{p}[/green]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path):
    """Write results list to CSV, creating headers from first row."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _print_matrix_summary(results: list[dict]):
    """Print a compact Rich summary table of all benchmark results."""
    table = Table(title="📊 Benchmark Matrix Summary", header_style="bold magenta", show_lines=True)
    cols = [
        ("Config Label",    "config_label"),
        ("Profile",         "load_profile"),
        ("Status",          "status"),
        ("RPS",             "throughput_rps"),
        ("P95 (ms)",        "p95_latency_ms"),
        ("$/1K Req",        "cost_per_1k_requests_usd"),
        ("Quality (0-1)",   "composite_quality_score"),
    ]
    for label, _ in cols:
        table.add_column(label, justify="right" if label not in ("Config Label", "Profile", "Status") else "left")

    for r in results:
        status_color = "green" if r.get("status") == "success" else "red"
        table.add_row(
            r.get("config_label", "?"),
            r.get("load_profile", "?"),
            f"[{status_color}]{r.get('status', '?')}[/{status_color}]",
            f"{r.get('throughput_rps', '?')}",
            f"{r.get('p95_latency_ms', '?')}",
            f"${r.get('cost_per_1k_requests_usd', 0):.6f}" if r.get("cost_per_1k_requests_usd") else "N/A",
            f"{r.get('composite_quality_score', '?')}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
