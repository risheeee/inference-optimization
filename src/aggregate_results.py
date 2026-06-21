"""
aggregate_results.py — Build benchmark_matrix.csv from manually-run results
===========================================================================
Run this when you've executed load_generator, cost_calculator, and
quality_evaluator manually (not via benchmark_runner sweep).

It reads:
  • results/cost_comparison.json   → throughput, latency, cost metrics
  • results/quality_*.json         → quality scores per engine

And writes:
  • results/benchmark_matrix_{ts}.csv  → readable by visualize_results.ipynb

Usage:
    python src/aggregate_results.py
"""
import csv
import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))

# ── Per-engine metadata ────────────────────────────────────────────────────────
ENGINE_META = {
    "vllm": {
        "weight_format":       "awq_4bit",
        "weight_format_label": "AWQ 4-bit (vLLM)",
        "kv_cache":            "fp16_kvcache",
        "kv_cache_label":      "KV Cache FP16",
        "batch_strategy":      "continuous",
        "batch_label":         "Continuous Batching",
    },
    "llamacpp": {
        "weight_format":       "gguf_q4",
        "weight_format_label": "GGUF Q4 (llama.cpp)",
        "kv_cache":            "fp16_kvcache",
        "kv_cache_label":      "KV Cache FP16",
        "batch_strategy":      "static",
        "batch_label":         "Static Batching",
    },
}


def load_cost_comparison():
    path = RESULTS_DIR / "cost_comparison.json"
    if not path.exists():
        print(f"⚠  No cost_comparison.json found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def load_quality_reports():
    """Return {engine_name: quality_data} using the most-recent file per engine."""
    by_engine = {}
    for fp in sorted(RESULTS_DIR.glob("quality_*.json")):
        with open(fp) as f:
            data = json.load(f)
        eng = data.get("engine", "unknown")
        by_engine[eng] = data   # sorted oldest→newest, so last write wins
    return by_engine


def infer_profile(filename: str) -> str:
    for p in ("short_qa", "medium_reasoning", "long_context", "burst_spike"):
        if p in filename:
            return p
    return "unknown"


def infer_engine(filename: str) -> str:
    if filename.startswith("vllm"):
        return "vllm"
    if filename.startswith("llamacpp") or filename.startswith("llama"):
        return "llamacpp"
    if filename.startswith("mock"):
        return "mock"
    return "unknown"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    costs = load_cost_comparison()
    quality_by_engine = load_quality_reports()

    print(f"📂 Found {len(costs)} cost entries")
    print(f"📂 Found quality reports for engines: {list(quality_by_engine.keys())}")

    # ── Deduplicate: keep best (highest throughput) entry per engine+profile ──
    best: dict = {}
    skipped = 0
    for entry in costs:
        if "error" in entry:
            skipped += 1
            print(f"  ⏭  Skipping errored entry: {entry.get('source_file')}")
            continue
        src = entry.get("source_file", "")
        engine = infer_engine(src)
        profile = infer_profile(src)
        if engine == "mock":
            print(f"  ⏭  Skipping mock entry: {src}")
            continue
        key = (engine, profile)
        existing = best.get(key)
        cur_rps = entry.get("throughput_rps") or 0
        old_rps = (existing or {}).get("throughput_rps") or 0
        if existing is None or cur_rps > old_rps:
            best[key] = entry

    print(f"  ℹ  {skipped} errored entries skipped, {len(best)} unique engine/profile combos kept")

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for (engine, profile), entry in sorted(best.items()):
        meta = ENGINE_META.get(engine, ENGINE_META["vllm"])
        config_label = (
            f"{engine}__{meta['weight_format']}"
            f"__{meta['kv_cache']}__{meta['batch_strategy']}"
        )

        # Quality: prefer same engine; fall back to vllm (same base model, close quant tier)
        qdata = quality_by_engine.get(engine) or quality_by_engine.get("vllm") or {}
        je    = qdata.get("json_extraction", {})
        rs    = qdata.get("reasoning", {})
        q_src = "measured" if engine in quality_by_engine else "estimated_from_vllm"

        row = {
            "engine":               engine,
            "weight_format":        meta["weight_format"],
            "weight_format_label":  meta["weight_format_label"],
            "kv_cache":             meta["kv_cache"],
            "kv_cache_label":       meta["kv_cache_label"],
            "batch_strategy":       meta["batch_strategy"],
            "batch_label":          meta["batch_label"],
            "load_profile":         profile,
            "config_label":         config_label,
            "timestamp":            datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status":               "success",
            # Throughput / latency
            "throughput_rps":           entry.get("throughput_rps"),
            "p50_latency_ms":           entry.get("total_latency_p50_ms"),
            "p95_latency_ms":           entry.get("total_latency_p95_ms"),
            "p99_latency_ms":           entry.get("total_latency_p99_ms"),
            "mean_ttft_ms":             entry.get("ttft_mean_ms"),
            "avg_tokens_per_second":    entry.get("aggregate_tokens_per_second"),
            # Cost
            "cost_per_1k_requests_usd": entry.get("cost_per_1k_requests_usd"),
            "cost_per_1m_tokens_usd":   entry.get("cost_per_1m_tokens_usd"),
            "gpu_efficiency_score":     entry.get("gpu_efficiency_score"),
            "error_rate_pct":           entry.get("error_rate_pct", 0.0),
            # Quality
            "json_accuracy_pct":        je.get("accuracy_pct"),
            "reasoning_score_avg":      rs.get("avg_score_1_to_5"),
            "composite_quality_score":  qdata.get("composite_quality_score_0_to_1"),
            "quality_source":           q_src,
        }
        rows.append(row)
        print(f"  ✓ {engine}/{profile} — RPS={row['throughput_rps']}, "
              f"P95={row['p95_latency_ms']}ms, "
              f"$/1K={row['cost_per_1k_requests_usd']}, "
              f"quality={row['composite_quality_score']} ({q_src})")

    if not rows:
        print("\n⚠  No rows to write. Check your results/ directory.")
        return

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"benchmark_matrix_{ts}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Written {len(rows)} rows to: {out}")
    print(f"   Now open notebooks/visualize_results.ipynb and run all cells.")


if __name__ == "__main__":
    main()
