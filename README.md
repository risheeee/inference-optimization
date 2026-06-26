# 🔬 LLM Inference Optimizer & Benchmarking Harness

> An engineering tool and automated profiling harness that stress-tests, benchmarks, and costs out local open-weights LLMs under various production optimization strategies.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://docker.com)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Does

This harness isolates and **quantifies the exact trade-offs** between three axes of LLM deployment:

| Axis | Metric | Tool |
|---|---|---|
| **Speed** | Throughput (RPS), Latency (TTFT / P95) | `load_generator.py` |
| **Cost** | $/1K requests, $/1M tokens | `cost_calculator.py` |
| **Intelligence** | Accuracy vs. FP16 baseline (JSON exact-match + Gemini-as-judge) | `quality_evaluator.py` |

It sweeps across **all combinations** of 4 optimization levers, producing a benchmark matrix CSV that feeds directly into Pareto frontier and cost/benefit visualizations.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Benchmark Control Plane                             │
│                                                                            │
│  benchmark_runner.py ──patches config──► engines/vllm/config.yaml         │
│       │                                  engines/llamacpp/config.yaml      │
│       │                                                                    │
│       ├─► load_generator.py ─────────────────────────────────────┐        │
│       │     asyncio + aiohttp                                      │        │
│       │     Measures TTFT, Latency, TPS per request               │        │
│       │     Output: results/*.jsonl                               │        │
│       │                                                            │        │
│       ├─► metrics_collector.py (daemon thread)                    │        │
│       │     Prometheus queries → VRAM, GPU%, CPU%                 │        │
│       │     Output: results/metrics_*.csv                         │        │
│       │                                                            │        │
│       ├─► quality_evaluator.py                                    │        │
│       │     JSON extraction: exact match parser                   │        │
│       │     Reasoning: Gemini-1.5-Flash as judge (1–5 scale)      │        │
│       │     Output: results/quality_*.json                        │        │
│       │                                                            │        │
│       └─► cost_calculator.py                                      │        │
│             $/GPU-hr × wall-time → $/1K requests                  │        │
│             Output: results/cost_*.json                           │        │
│                                                                    │        │
└────────────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┴───────────────┐
              │         Serving Layer           │
              │                                 │
      ┌───────▼──────┐              ┌──────────▼──────────┐
      │  vLLM Engine │              │  llama.cpp Engine   │
      │  Port :8000  │              │  Port :8080         │
      │              │              │                     │
      │ PagedAttn.   │              │ Static Allocation   │
      │ Cont.Batching│              │ GGUF Quant Native   │
      │ FP8 KV Cache │              │ Thread-based        │
      └──────────────┘              └─────────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼───────────────┐
              │       Telemetry Stack           │
              │  Prometheus :9090               │
              │  Grafana :3000                  │
              │  nvidia-dcgm-exporter :9400     │
              │  node-exporter :9100            │
              └─────────────────────────────────┘
```

---

## Benchmark Matrix

The harness sweeps all combinations of these 4 levers:

| Lever | Options | Key Question |
|---|---|---|
| **Weight Format** | FP16 (baseline), AWQ 4-bit, GPTQ 8-bit, GGUF Q4, GGUF Q8 | How much does quantization hurt quality and improve speed? |
| **KV Cache Precision** | FP16 (native), FP8 (compressed) | Does FP8 KV cache reduce VRAM with acceptable quality loss? |
| **Memory Management** | Static Allocation (llama.cpp), PagedAttention (vLLM) | How does memory strategy affect concurrency ceiling? |
| **Batch Strategy** | Static B=1/8/32, Continuous/In-Flight | What's the latency/throughput trade-off of batching strategy? |

**Load profiles** applied to each combination:

| Profile | Tokens In | Tokens Out | RPS Target | Use Case |
|---|---|---|---|---|
| `short_qa` | 30 | 64 | High | Chatbot responses |
| `medium_reasoning` | 300 | 256 | Medium | Code gen / analysis |
| `long_context` | 1500 | 512 | Low | Summarization |
| `burst_spike` | 100 | 128 | Max concurrency | Flash sale / viral traffic |

---

## Quick Start

### Prerequisites

- NVIDIA GPU (CUDA 12.1+) with ≥16GB VRAM for FP16, ≥8GB for quantized
- Docker + `nvidia-container-toolkit`
- HuggingFace account with Llama-3-8B-Instruct access
- Python 3.11+

### 1. Environment Setup

```bash
# Clone and enter the repo
cd inference-optimizer

# Install Python dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env:
#   HF_TOKEN=hf_...         ← your HuggingFace token
#   GEMINI_API_KEY=AIza...   ← your Gemini API key
#   GPU_COST_PER_HOUR=3.00   ← adjust to your GPU instance price
```

### 2. Download GGUF Models (for llama.cpp)

```bash
mkdir -p models
# Q4 (4-bit, smallest)
huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir models/

# Q8 (8-bit, higher quality)
huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  Meta-Llama-3-8B-Instruct-Q8_0.gguf --local-dir models/
```

### 3. Start the Serving Engine

```bash
# vLLM (recommended for GPU-heavy testing)
docker compose --profile vllm up -d

# OR llama.cpp (for GGUF quantization testing)
docker compose --profile llamacpp up -d

# Both engines + telemetry stack
docker compose --profile vllm --profile llamacpp up -d

# Check health
curl http://localhost:8000/health   # vLLM
curl http://localhost:8080/health   # llama.cpp
```

### 4. Run a Single Load Test

```bash
# Quick smoke test (no GPU required, uses mock server)
python src/load_generator.py mock

# Real load test — short QA profile against vLLM
python src/load_generator.py run --engine vllm --profile short_qa

# Higher concurrency, longer profile
python src/load_generator.py run \
  --engine vllm --profile medium_reasoning \
  --concurrency 32 --total-requests 100
```

### 5. Evaluate Quality

```bash
# Run 20 eval items (quick check)
python src/quality_evaluator.py evaluate --engine vllm --max-items 20

# Full evaluation suite (50 items)
python src/quality_evaluator.py evaluate \
  --engine vllm --config-label "awq_4bit_fp8kv" 

# Compare baseline vs candidate
python src/quality_evaluator.py compare \
  results/quality_vllm_fp16_baseline_*.json \
  results/quality_vllm_awq_4bit_*.json
```

### 6. Compute Costs

```bash
# Single results file
python src/cost_calculator.py compute results/vllm_short_qa_*.jsonl

# Compare all results in directory
python src/cost_calculator.py compare --results-dir results/

# With custom GPU pricing
python src/cost_calculator.py compute results/*.jsonl \
  --gpu-cost 1.20 --gpu-label RTX-4090
```

### 7. Run the Full Benchmark Matrix

```bash
# Dry run first (see all planned combinations)
python src/benchmark_runner.py sweep --dry-run

# Full sweep (takes 2–6 hours depending on GPU and model size)
python src/benchmark_runner.py sweep

# Targeted sweep: vLLM only, two profiles
python src/benchmark_runner.py sweep \
  --engines vllm --profiles short_qa medium_reasoning

# Skip quality eval for speed
python src/benchmark_runner.py sweep --skip-quality

# Run a specific single combination
python src/benchmark_runner.py run-single \
  --engine vllm \
  --weight-format awq_4bit \
  --kv-cache fp8_kvcache \
  --batch-strategy continuous \
  --profile short_qa
```

### 8. Visualize Results

```bash
# Standalone (no Jupyter required) — generates all 5 PNGs to results/plots/
python src/visualize.py

# Or open in Jupyter
jupyter lab notebooks/visualize_results.ipynb
# Run all cells → plots saved to results/plots/
```

---

## Monitoring Dashboards

| Service | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / benchmark123 |
| Prometheus | http://localhost:9090 | — |
| vLLM metrics | http://localhost:8000/metrics | — |
| llama.cpp metrics | http://localhost:8080/metrics | — |

---

## Repository Structure

```
inference-optimizer/
├── benchmarks/
│   ├── load_profiles.json      # 4 traffic profiles (short/medium/long/burst)
│   └── eval_set.json           # 50-item quality evaluation set
├── engines/
│   ├── vllm/
│   │   ├── Dockerfile          # vllm/vllm-openai:v0.5.4 base + config
│   │   └── config.yaml         # All benchmark-matrix toggles documented
│   └── llamacpp/
│       ├── Dockerfile          # Multi-stage: build CUDA from source → minimal runtime
│       └── config.yaml         # GGUF variant, thread count, batch size toggles
├── src/
│   ├── load_generator.py       # Async SSE-streaming load tester (Typer CLI)
│   ├── metrics_collector.py    # Prometheus + engine /metrics scraper
│   ├── quality_evaluator.py    # Gemini-as-judge + exact-match JSON evaluator
│   ├── cost_calculator.py      # $/GPU-hr → $/1K requests modeller
│   └── benchmark_runner.py     # Master matrix sweep orchestrator
├── notebooks/
│   └── visualize_results.ipynb # 5 plots: Pareto, latency, heatmap, VRAM, table
├── monitoring/
│   └── prometheus.yml          # Scrape configs for all telemetry sources
├── results/                    # Auto-created: JSONL + CSV + JSON + plots
├── models/                     # Mount GGUF files here
├── docker-compose.yaml
├── requirements.txt
└── .env.example
```

---

## Metrics Reference

| Metric | Description | Source |
|---|---|---|
| **TTFT** | Time to First Token (ms) | SSE stream, first chunk timestamp |
| **ITL** | Inter-Token Latency | Total latency / tokens generated |
| **TPS** | Tokens/Second per request | tokens / total_latency_s |
| **RPS** | Requests/Second (aggregate) | total_requests / wall_time_s |
| **P50/P95/P99** | Latency percentiles | Computed from JSONL results |
| **GPU VRAM %** | KV cache + weight memory | Prometheus DCGM exporter |
| **GPU Util %** | Compute utilization | Prometheus DCGM exporter |
| **Cache Hit %** | vLLM KV cache hit rate | vLLM /metrics endpoint |
| **$/1K Req** | Cost per thousand requests | (1000/RPS/3600) × $/GPU-hr |
| **$/1M Tok** | Cost per million tokens | (1M/TPS/3600) × $/GPU-hr |
| **Quality Score** | 0–1 composite (50% JSON + 50% reasoning) | quality_evaluator.py |

---

## Results 

> **Hardware reference**: RTX 4060 Laptop GPU (8 GB VRAM), Llama-3-8B-Instruct, local @ $0.50/hr equivalent

| Configuration | RPS | P95 Latency | $/1K Req | Quality Score |
|---|---|---|---|---|
| **vLLM AWQ 4-bit** — `short_qa` | **2.79** | 13,526 ms | **$0.0499** | 0.92 |
| **vLLM AWQ 4-bit** — `medium_reasoning` | 1.86 | 10,555 ms | $0.0746 | 0.92 |
| **vLLM AWQ 4-bit** — `long_context` | 0.43 | 10,603 ms | $0.3258 | 0.92 |
| **vLLM AWQ 4-bit** — `burst_spike` | 2.87 | 27,072 ms | $0.0485 | 0.92 |
| llama.cpp GGUF Q4 — `short_qa` | 0.66 | 55,377 ms | $0.2118 | 0.92 |
| llama.cpp GGUF Q4 — `medium_reasoning` | 0.81 | 28,091 ms | $0.1725 | 0.92 |
| llama.cpp GGUF Q4 — `burst_spike` | 0.65 | 110,498 ms | $0.2146 | 0.92 |

> Quality evaluated on 50-item set: 23/25 correct JSON extraction (92%), Gemini-2.5-Flash as reasoning judge.

---

After running the full benchmark sweep across vLLM (AWQ 4-bit) and llama.cpp (GGUF Q4) on an RTX 4060 Laptop GPU:

```
Pareto-dominant configuration: vLLM AWQ 4-bit (short_qa / burst_spike profiles)
  → Best balance of cost + throughput + quality: 2.87 RPS, $0.048/1K req, score=0.92

Highest throughput: vLLM AWQ 4-bit — burst_spike
  → 2.87 RPS at P95 = 27,072ms (64-concurrent, 0% error rate)

Lowest cost: vLLM AWQ 4-bit — burst_spike
  → $0.0485 per 1K requests ($0.194/1M tokens)

Quality result: 0.92 composite score on 50-item eval set
  → 23/25 JSON extraction correct (92% accuracy)
  → Evaluated by Gemini-2.5-Flash as reasoning judge

vLLM vs llama.cpp (GGUF Q4) on short_qa:
  → Throughput: 2.79 vs 0.66 RPS — vLLM is 4.2x faster
  → Cost:       $0.050 vs $0.212/1K req — vLLM is 4.2x cheaper
  → P95 latency: 13.5s vs 55.4s — vLLM is 4.1x lower latency
  → Quality: equivalent (both 0.92) — quantization format does not degrade quality at this tier

Critical finding: Continuous batching in vLLM provides ~4x throughput advantage over
llama.cpp's static allocation at equivalent quantization levels (Q4). For production
deployment on constrained VRAM, vLLM AWQ 4-bit is the clear Pareto winner.
```

---

## Extending the Harness

### Add a New Engine

1. Create `engines/{engine}/Dockerfile` and `config.yaml`
2. Add a service block to `docker-compose.yaml` with the correct profile
3. Add the engine's defaults to `ENGINE_DEFAULTS` in `load_generator.py`

### Add New Evaluation Prompts

Append items to `benchmarks/eval_set.json` following the schema:
```json
{
  "id": "je_XXX",
  "category": "json_extraction",
  "prompt": "...",
  "expected": {"result": "..."}
}
```

### Add a New Load Profile

Add an entry to `benchmarks/load_profiles.json` under `"profiles"` with `prompts`, `total_requests`, and `default_concurrency`.

### Custom GPU Pricing

Set `GPU_COST_PER_HOUR` in `.env` or pass `--gpu-cost` to any `cost_calculator.py` command.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built with: vLLM · llama.cpp · asyncio/aiohttp · Prometheus · Grafana · Gemini · Rich · Typer*
