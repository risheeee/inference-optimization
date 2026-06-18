# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp Serving Engine — Dockerfile
# Builds llama.cpp server from source with CUDA support (GGML_CUDA=1).
# Exposes an OpenAI-compatible /v1/* API on port 8080.
# GGUF model weights must be mounted at /models/ via docker volume.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build llama.cpp with CUDA ───────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Suppress interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp at a pinned commit for reproducibility
ARG LLAMACPP_VERSION=b3712
RUN git clone --depth 1 --branch ${LLAMACPP_VERSION} \
    https://github.com/ggerganov/llama.cpp /llama.cpp || \
    git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama.cpp

WORKDIR /llama.cpp

# Build with CUDA support, optimized for runtime architecture
# GGML_CUDA_F16=1 enables FP16 CUDA kernels for faster inference
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    && cmake --build build --config Release --target llama-server -j$(nproc)

# ── Stage 2: Minimal Runtime Image ───────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub for optional model download at runtime
RUN pip3 install --no-cache-dir huggingface_hub[cli]>=0.23.0

# Copy compiled server binary from builder stage
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
RUN chmod +x /usr/local/bin/llama-server

# Copy config
COPY config.yaml /app/config.yaml

# Model weights directory (mount external volume here)
RUN mkdir -p /models

WORKDIR /app

EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Model path, batch size, and thread count are passed via docker-compose
# environment variables or directly as CMD overrides.
ENTRYPOINT ["llama-server"]
CMD [ \
    "--model",        "/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", \
    "--alias",        "llama-3-8b-instruct", \
    "--ctx-size",     "4096", \
    "--n-gpu-layers", "35", \
    "--threads",      "8", \
    "--batch-size",   "512", \
    "--parallel",     "8", \
    "--host",         "0.0.0.0", \
    "--port",         "8080", \
    "--log-format",   "json" \
]

# ── Build-time labels ─────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="llama.cpp Benchmark Engine"
LABEL org.opencontainers.image.description="llama.cpp server (CUDA) for LLM inference benchmarking"
LABEL benchmark.engine="llamacpp"
LABEL benchmark.cuda_build=true
