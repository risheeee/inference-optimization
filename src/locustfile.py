"""
locustfile.py — Distributed Load Testing with Locust
=====================================================
IMPORTANT: Locust must be imported FIRST in this file.
Locust uses gevent which monkey-patches ssl at import time.
Importing it first (before aiohttp or requests) avoids the MonkeyPatchWarning.

Run with:
  # Headless (CI / scripted)
  locust -f src/locustfile.py --headless -u 32 -r 4 --run-time 60s \
    --host http://localhost:8000

  # With web UI at http://localhost:8089
  locust -f src/locustfile.py --host http://localhost:8000

  # Against llama.cpp
  locust -f src/locustfile.py --headless -u 8 -r 2 --run-time 60s \
    --host http://localhost:8080
"""

# ── gevent must be monkey-patched FIRST ──────────────────────────────────────
# Locust's __init__.py calls monkey.patch_all() which patches ssl, socket, etc.
# All other imports must come AFTER this.
from locust import HttpUser, between, task, events  # noqa: E402 (must be first)

import json
import os
import random
import time
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
VLLM_MODEL = os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL_NAME", "llama-3-8b-instruct")

SHORT_QA_PROMPTS = [
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is 17 multiplied by 23?",
    "What is the chemical symbol for gold?",
    "Who invented the telephone?",
    "What does HTTP stand for?",
    "What is the smallest prime number?",
    "What is the boiling point of water in Celsius?",
    "What is the largest planet in our solar system?",
    "Who developed the theory of general relativity?",
]

MEDIUM_PROMPTS = [
    "Explain step-by-step how a transformer's self-attention mechanism works.",
    "What are the SOLID principles in object-oriented design? Explain each briefly.",
    "Explain the difference between horizontal and vertical scaling with trade-offs.",
    "Describe the TCP three-way handshake and why each step is necessary.",
    "Compare SQL vs NoSQL databases with concrete use-case examples.",
    "Explain gradient descent and common variants: SGD, Adam, RMSProp.",
    "What is the CAP theorem and what trade-offs does it force on engineers?",
    "Explain mutex vs semaphore with a concrete real-world example.",
]


# ══════════════════════════════════════════════════════════════════════════════
# User Classes
# ══════════════════════════════════════════════════════════════════════════════

class ShortQAUser(HttpUser):
    """
    High-concurrency short QA user.
    Simulates many concurrent users asking short questions.
    Targets: /v1/completions (OpenAI-compatible)
    """
    wait_time = between(0.05, 0.3)

    @task(3)
    def short_completion(self):
        """Short prompt, 64 token response — tests burst throughput."""
        prompt = random.choice(SHORT_QA_PROMPTS)
        payload = {
            "model": VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
        }
        start = time.perf_counter()
        with self.client.post(
            "/v1/completions",
            json=payload,
            catch_response=True,
            name="POST /v1/completions [short_qa]",
        ) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    resp.success()
                except Exception:
                    resp.failure("Invalid JSON response")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")

    @task(1)
    def health_check(self):
        """Lightweight health check — validates engine is alive."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")


class MediumReasoningUser(HttpUser):
    """
    Medium-concurrency reasoning user.
    Sends longer prompts with more complex outputs.
    """
    wait_time = between(0.5, 2.0)

    @task
    def reasoning_completion(self):
        """Medium prompt, 256 token response — tests sustained throughput."""
        prompt = random.choice(MEDIUM_PROMPTS)
        payload = {
            "model": VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
        }
        with self.client.post(
            "/v1/completions",
            json=payload,
            catch_response=True,
            name="POST /v1/completions [reasoning]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


class BurstSpikeUser(HttpUser):
    """
    Maximum concurrency burst user.
    No wait time — hammers the endpoint as fast as possible.
    Use sparingly: locust -u 64 -r 64 --run-time 30s
    """
    wait_time = between(0, 0.05)

    @task
    def burst_completion(self):
        prompt = random.choice(SHORT_QA_PROMPTS)
        payload = {
            "model": VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.0,
            "stream": False,
        }
        with self.client.post(
            "/v1/completions",
            json=payload,
            catch_response=True,
            name="POST /v1/completions [burst]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# Locust Event Hooks
# ══════════════════════════════════════════════════════════════════════════════

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"\n🚀 Locust test started — target: {environment.host}")
    print(f"   Model: {VLLM_MODEL}")
    print(f"   User classes: {[u.__name__ for u in environment.user_classes]}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    print(f"\n📊 Test complete — {stats.num_requests} requests, "
          f"fail rate: {stats.fail_ratio * 100:.1f}%, "
          f"median: {stats.median_response_time}ms, "
          f"p95: {stats.get_response_time_percentile(0.95)}ms")
