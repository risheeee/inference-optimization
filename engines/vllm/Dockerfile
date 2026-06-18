# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# vLLM Serving Engine — Dockerfile
# Base: official vLLM OpenAI-compatible server with CUDA 12.1 + PyTorch 2.3
# Exposes an OpenAI-compatible /v1/* API on port 8000.
# Model weights are NOT bundled — mount via volume or pass HF_TOKEN env var.
# ─────────────────────────────────────────────────────────────────────────────
FROM vllm/vllm-openai:v0.5.4

# Install huggingface_hub CLI for on-demand model downloads
RUN pip install --no-cache-dir huggingface_hub[cli]>=0.23.0

# Copy our engine config
COPY config.yaml /app/config.yaml

# Working directory
WORKDIR /app

# Expose the OpenAI-compatible server port
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
# The config.yaml values are passed as CLI flags.
# docker-compose overrides specific flags via environment variables.
# HF_TOKEN must be set in the environment or .env file for gated models.
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD [ \
    "--model",                    "meta-llama/Meta-Llama-3-8B-Instruct", \
    "--dtype",                    "auto", \
    "--kv-cache-dtype",           "auto", \
    "--gpu-memory-utilization",   "0.90", \
    "--max-model-len",            "4096", \
    "--max-num-seqs",             "32", \
    "--enable-chunked-prefill", \
    "--host",                     "0.0.0.0", \
    "--port",                     "8000" \
]

# ── Build-time labels ─────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="vLLM Benchmark Engine"
LABEL org.opencontainers.image.description="vLLM OpenAI-compatible server for LLM inference benchmarking"
LABEL benchmark.engine="vllm"
LABEL benchmark.default_model="meta-llama/Meta-Llama-3-8B-Instruct"
