"""
quality_evaluator.py — LLM Output Quality Grader
=================================================
Evaluates whether inference optimizations (quantization, KV cache compression,
etc.) have degraded the model's reasoning capabilities vs. the FP16 baseline.

Two evaluation strategies:

  1. json_extraction items — Exact-match JSON parsing:
       • Sends the prompt to the model under test.
       • Attempts to parse the response as JSON and checks if
         the "result" key matches the expected value (type-flexible).
       • Score: 1.0 (correct) or 0.0 (wrong/invalid JSON)

  2. reasoning items — Gemini-as-judge (1–5 scale):
       • Sends the original prompt + model response to Gemini Flash.
       • Gemini scores the response on correctness and quality (1–5).
       • Falls back to keyword-matching heuristics if Gemini is unavailable.

Outputs:
  • Per-item detailed scores
  • Aggregate report: json_accuracy_pct, reasoning_score_avg, overall_score
  • JSON report file: results/quality_{engine}_{config}_{timestamp}.json

Usage:
  # Run evaluation against a live engine endpoint
  python src/quality_evaluator.py evaluate --engine vllm

  # Grade a pre-collected results JSONL file
  python src/quality_evaluator.py grade --results results/vllm_short_qa_xyz.jsonl

  # Compare two configurations
  python src/quality_evaluator.py compare \\
    --baseline results/quality_vllm_fp16_baseline.json \\
    --candidate results/quality_vllm_awq4bit.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import asyncio
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
EVAL_SET_PATH = Path(__file__).parent.parent / "benchmarks" / "eval_set.json"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-1.5-flash")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080")
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
JUDGE_CONCURRENCY = 5  # concurrent Gemini judge calls

console = Console()
app = typer.Typer(
    name="quality-evaluator",
    help="LLM output quality grader using Gemini-as-judge.",
    add_completion=False,
)


# ══════════════════════════════════════════════════════════════════════════════
# Gemini Judge Client
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating the quality of an AI language model's response to a question.

You will be given:
- QUESTION: The original prompt given to the model
- REFERENCE: The expected correct answer or key reasoning steps
- RESPONSE: The model's actual response

Score the RESPONSE on a scale of 1-5:
5 = Fully correct, well-reasoned, clearly explains each step
4 = Mostly correct with minor errors or omissions
3 = Partially correct, key concepts present but significant gaps
2 = Major errors but shows some relevant understanding  
1 = Completely wrong, incoherent, or refuses to answer

Respond with ONLY a JSON object in this format: {"score": <integer 1-5>, "reason": "<one sentence explanation>"}
Do not include any other text."""


async def gemini_judge_score(
    session: aiohttp.ClientSession,
    question: str,
    reference: str,
    response: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Call Gemini Flash to score a model response on a 1-5 scale.
    Returns {"score": int, "reason": str} or {"score": None, "reason": "error msg"}.
    """
    if not GEMINI_API_KEY:
        return {"score": None, "reason": "GEMINI_API_KEY not set — using heuristic fallback"}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_JUDGE_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"MODEL RESPONSE:\n{response}"
    )

    payload = {
        "system_instruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 128,
            "responseMimeType": "application/json",
        },
    }

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    data = await resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    parsed = json.loads(text)
                    score = int(parsed.get("score", 0))
                    if not (1 <= score <= 5):
                        raise ValueError(f"Score out of range: {score}")
                    return {"score": score, "reason": parsed.get("reason", "")}
            except Exception as exc:
                if attempt == 2:
                    return {"score": None, "reason": f"Judge error: {exc}"}
                await asyncio.sleep(1)

    return {"score": None, "reason": "All judge attempts failed"}


def heuristic_reasoning_score(response: str, expected_answer: str) -> dict:
    """
    Fallback scorer when Gemini is unavailable.
    Does keyword matching and returns a rough 1-5 score.
    """
    if not response or len(response.strip()) < 10:
        return {"score": 1, "reason": "Empty or trivial response (heuristic)"}

    # Extract key numbers/phrases from expected answer
    expected_lower = expected_answer.lower()
    response_lower = response.lower()

    # Find numbers in expected answer
    expected_numbers = set(re.findall(r'\d+\.?\d*', expected_answer))
    response_numbers = set(re.findall(r'\d+\.?\d*', response))
    number_overlap = len(expected_numbers & response_numbers) / max(len(expected_numbers), 1)

    # Find key words (filter stopwords roughly)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in", "at"}
    expected_words = set(w for w in expected_lower.split() if w not in stopwords and len(w) > 3)
    response_words = set(w for w in response_lower.split() if w not in stopwords and len(w) > 3)
    word_overlap = len(expected_words & response_words) / max(len(expected_words), 1) if expected_words else 0

    combined = 0.5 * number_overlap + 0.5 * word_overlap
    score = max(1, min(5, round(1 + combined * 4)))
    return {"score": score, "reason": f"Heuristic (num_overlap={number_overlap:.2f}, word_overlap={word_overlap:.2f})"}


# ══════════════════════════════════════════════════════════════════════════════
# JSON Extraction Evaluator
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_json_extraction(response: str, expected: dict) -> dict:
    """
    Evaluate a JSON extraction response.
    Returns {"score": float, "correct": bool, "parsed": ..., "reason": str}
    """
    # Try to find JSON in the response
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if not json_match:
        return {"score": 0.0, "correct": False, "parsed": None, "reason": "No JSON found in response"}

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"score": 0.0, "correct": False, "parsed": None, "reason": f"Invalid JSON: {e}"}

    # Check if "result" key exists
    if "result" not in parsed:
        return {"score": 0.0, "correct": False, "parsed": parsed, "reason": "Missing 'result' key in JSON"}

    actual = parsed["result"]
    exp = expected.get("result")

    # Type-flexible comparison
    def normalize(v):
        if isinstance(v, str):
            return v.strip().lower()
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v

    if normalize(actual) == normalize(exp):
        return {"score": 1.0, "correct": True, "parsed": parsed, "reason": "Exact match"}

    # Check string representations
    if str(actual).strip().lower() == str(exp).strip().lower():
        return {"score": 1.0, "correct": True, "parsed": parsed, "reason": "String match"}

    # Partial credit: numeric within tolerance
    try:
        if abs(float(actual) - float(exp)) < 0.01:
            return {"score": 1.0, "correct": True, "parsed": parsed, "reason": "Numeric match within tolerance"}
    except (ValueError, TypeError):
        pass

    return {
        "score": 0.0, "correct": False, "parsed": parsed,
        "reason": f"Wrong value: got '{actual}', expected '{exp}'"
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model Inference Client
# ══════════════════════════════════════════════════════════════════════════════

async def query_engine(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Send a single non-streaming completion request and return the response text."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": DEFAULT_TEMPERATURE,
        "stream": False,
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
            return data["choices"][0]["text"].strip()
    except Exception as e:
        return f"[ERROR: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# Main Evaluation Loop
# ══════════════════════════════════════════════════════════════════════════════

async def run_evaluation(
    engine: str,
    base_url: str,
    model: str,
    eval_items: list[dict],
    config_label: str = "default",
    use_gemini_judge: bool = True,
) -> dict:
    """
    Run full evaluation suite against the specified engine.
    Returns a complete quality report dict.
    """
    judge_semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
    item_results = []

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:

        with Progress(
            SpinnerColumn(),
            "[cyan]{task.description}",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(f"[cyan]Evaluating {len(eval_items)} items...", total=len(eval_items))

            for item in eval_items:
                category = item["category"]
                prompt = item["prompt"]

                # Query the model
                response = await query_engine(session, base_url, model, prompt)

                result = {
                    "id": item["id"],
                    "category": category,
                    "prompt_preview": prompt[:100],
                    "response_preview": response[:200],
                    "full_response": response,
                }

                if category == "json_extraction":
                    eval_result = evaluate_json_extraction(response, item.get("expected", {}))
                    result.update({
                        "score": eval_result["score"],
                        "correct": eval_result["correct"],
                        "parsed_json": str(eval_result.get("parsed")),
                        "reason": eval_result["reason"],
                        "method": "exact_match_json",
                    })

                elif category == "reasoning":
                    ref_answer = item.get("expected_answer", item.get("reference_reasoning", ""))
                    if use_gemini_judge and GEMINI_API_KEY:
                        judge_result = await gemini_judge_score(session, prompt, ref_answer, response, judge_semaphore)
                    else:
                        judge_result = heuristic_reasoning_score(response, ref_answer)

                    result.update({
                        "score": judge_result["score"] / 5.0 if judge_result["score"] else None,  # normalize to 0-1
                        "raw_judge_score": judge_result["score"],
                        "reason": judge_result["reason"],
                        "method": "gemini_judge" if (use_gemini_judge and GEMINI_API_KEY) else "heuristic",
                    })

                item_results.append(result)
                progress.advance(task_id)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    json_items = [r for r in item_results if r["category"] == "json_extraction"]
    reasoning_items = [r for r in item_results if r["category"] == "reasoning"]

    json_scores = [r["score"] for r in json_items if r.get("score") is not None]
    reasoning_scores = [r["raw_judge_score"] for r in reasoning_items if r.get("raw_judge_score")]

    json_accuracy_pct = (sum(json_scores) / len(json_scores) * 100) if json_scores else None
    reasoning_score_avg = (sum(reasoning_scores) / len(reasoning_scores)) if reasoning_scores else None
    # Composite: 50% JSON accuracy + 50% reasoning (normalized)
    composite_score = None
    if json_accuracy_pct is not None and reasoning_score_avg is not None:
        composite_score = round((json_accuracy_pct / 100 * 0.5) + (reasoning_score_avg / 5.0 * 0.5), 4)
    elif json_accuracy_pct is not None:
        composite_score = round(json_accuracy_pct / 100, 4)
    elif reasoning_score_avg is not None:
        composite_score = round(reasoning_score_avg / 5.0, 4)

    report = {
        "engine": engine,
        "config_label": config_label,
        "model": model,
        "base_url": base_url,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_items_evaluated": len(item_results),
        "json_extraction": {
            "count": len(json_items),
            "accuracy_pct": round(json_accuracy_pct, 2) if json_accuracy_pct is not None else None,
            "correct_count": int(sum(json_scores)) if json_scores else 0,
        },
        "reasoning": {
            "count": len(reasoning_items),
            "judge_model": GEMINI_JUDGE_MODEL if (use_gemini_judge and GEMINI_API_KEY) else "heuristic",
            "avg_score_1_to_5": round(reasoning_score_avg, 3) if reasoning_score_avg else None,
            "avg_score_normalized": round(reasoning_score_avg / 5.0, 4) if reasoning_score_avg else None,
        },
        "composite_quality_score_0_to_1": composite_score,
        "item_results": item_results,
    }
    return report


def print_quality_report(report: dict):
    """Render a quality report as a rich table."""
    table = Table(
        title=f"🎯 Quality Report — {report['engine']} / {report['config_label']}",
        show_header=True, header_style="bold yellow",
    )
    table.add_column("Metric", style="cyan", min_width=38)
    table.add_column("Value", style="white", justify="right")

    je = report.get("json_extraction", {})
    rs = report.get("reasoning", {})

    rows = [
        ("Engine", report.get("engine")),
        ("Config", report.get("config_label")),
        ("Total Items Evaluated", report.get("total_items_evaluated")),
        ("── JSON Extraction ──────────────────────────", None),
        ("  Count", je.get("count")),
        ("  Correct", je.get("correct_count")),
        ("  Accuracy (%)", f"{je.get('accuracy_pct'):.1f}%" if je.get("accuracy_pct") is not None else "N/A"),
        ("── Reasoning (Gemini Judge) ─────────────────", None),
        ("  Count", rs.get("count")),
        ("  Judge Model", rs.get("judge_model")),
        ("  Avg Score (1–5)", f"{rs.get('avg_score_1_to_5'):.3f}" if rs.get("avg_score_1_to_5") else "N/A"),
        ("  Avg Score (0–1)", f"{rs.get('avg_score_normalized'):.4f}" if rs.get("avg_score_normalized") else "N/A"),
        ("── Composite ────────────────────────────────", None),
        ("  Quality Score (0–1)", f"{report.get('composite_quality_score_0_to_1'):.4f}" if report.get("composite_quality_score_0_to_1") else "N/A"),
    ]
    for label, value in rows:
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

@app.command("evaluate")
def cmd_evaluate(
    engine: str = typer.Option("vllm", help="Engine to query: vllm | llamacpp"),
    config_label: str = typer.Option("default", help="Label for this config (e.g. 'awq_4bit_fp8_kvcache')"),
    max_items: Optional[int] = typer.Option(None, help="Limit number of eval items (for quick tests)"),
    base_url: Optional[str] = typer.Option(None, help="Override engine base URL"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    no_gemini: bool = typer.Option(False, help="Disable Gemini judge, use heuristic scoring"),
    output: Optional[Path] = typer.Option(None, help="Save JSON report to path"),
):
    """Run the full evaluation suite against a live engine."""
    if not EVAL_SET_PATH.exists():
        console.print(f"[red]eval_set.json not found at {EVAL_SET_PATH}[/red]")
        raise typer.Exit(1)

    with open(EVAL_SET_PATH) as f:
        eval_data = json.load(f)
    items = eval_data["items"]
    if max_items:
        items = items[:max_items]

    ENGINE_URLS = {"vllm": VLLM_BASE_URL, "llamacpp": LLAMACPP_BASE_URL}
    ENGINE_MODELS = {
        "vllm": os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
        "llamacpp": os.getenv("LLAMACPP_MODEL_NAME", "llama-3-8b-instruct"),
    }
    _base_url = base_url or ENGINE_URLS.get(engine, VLLM_BASE_URL)
    _model = model or ENGINE_MODELS.get(engine, "meta-llama/Meta-Llama-3-8B-Instruct")

    console.print(f"\n[bold cyan]🎯 Quality Evaluation[/bold cyan]")
    console.print(f"Engine: {engine} ({_base_url}), Items: {len(items)}")
    console.print(f"Judge: {'Gemini ' + GEMINI_JUDGE_MODEL if not no_gemini else 'Heuristic'}\n")

    report = asyncio.run(run_evaluation(
        engine=engine,
        base_url=_base_url,
        model=_model,
        eval_items=items,
        config_label=config_label,
        use_gemini_judge=not no_gemini,
    ))

    print_quality_report(report)

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output or RESULTS_DIR / f"quality_{engine}_{config_label}_{ts}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n[bold green]✓ Quality report saved to:[/bold green] {out}")


@app.command("compare")
def cmd_compare(
    baseline: Path = typer.Argument(..., help="Baseline quality report JSON (e.g. FP16)"),
    candidate: Path = typer.Argument(..., help="Candidate quality report JSON (e.g. AWQ 4-bit)"),
):
    """Compare quality degradation between two configurations."""
    with open(baseline) as f:
        b = json.load(f)
    with open(candidate) as f:
        c = json.load(f)

    table = Table(title="📉 Quality Degradation Analysis", header_style="bold red")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Baseline ({b.get('config_label', '?')})", justify="right")
    table.add_column(f"Candidate ({c.get('config_label', '?')})", justify="right")
    table.add_column("Delta", justify="right", style="yellow")

    def delta_fmt(b_val, c_val, pct=False):
        if b_val is None or c_val is None:
            return "N/A"
        d = c_val - b_val
        prefix = "+" if d >= 0 else ""
        if pct:
            return f"[{'green' if d >= 0 else 'red'}]{prefix}{d:.2f}pp[/]"
        return f"[{'green' if d >= 0 else 'red'}]{prefix}{d:.4f}[/]"

    b_json = b.get("json_extraction", {}).get("accuracy_pct")
    c_json = c.get("json_extraction", {}).get("accuracy_pct")
    b_reas = b.get("reasoning", {}).get("avg_score_1_to_5")
    c_reas = c.get("reasoning", {}).get("avg_score_1_to_5")
    b_comp = b.get("composite_quality_score_0_to_1")
    c_comp = c.get("composite_quality_score_0_to_1")

    table.add_row("JSON Accuracy (%)",
        f"{b_json:.1f}%" if b_json else "N/A",
        f"{c_json:.1f}%" if c_json else "N/A",
        delta_fmt(b_json, c_json, pct=True))
    table.add_row("Reasoning Score (1-5)",
        f"{b_reas:.3f}" if b_reas else "N/A",
        f"{c_reas:.3f}" if c_reas else "N/A",
        delta_fmt(b_reas, c_reas))
    table.add_row("Composite Quality (0-1)",
        f"{b_comp:.4f}" if b_comp else "N/A",
        f"{c_comp:.4f}" if c_comp else "N/A",
        delta_fmt(b_comp, c_comp))

    console.print()
    console.print(table)

    if b_comp and c_comp:
        degradation_pct = (b_comp - c_comp) / b_comp * 100
        color = "green" if degradation_pct < 2 else "yellow" if degradation_pct < 5 else "red"
        console.print(f"\n[bold {color}]Quality Degradation: {degradation_pct:+.2f}% vs baseline[/bold {color}]")


if __name__ == "__main__":
    app()
