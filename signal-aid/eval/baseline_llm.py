"""Baseline: the same four triage decisions made by an LLM on Amazon Bedrock.

    export BEDROCK_MODEL_ID=<a model id or inference profile from your Bedrock console>
    python -m eval.baseline_llm --data data/messages_sample.jsonl \
        --price-in 0.0 --price-out 0.0   # USD per 1M tokens, from the Bedrock pricing page

Writes eval/baseline.md and eval/baseline.json so you can compare accuracy,
latency and cost with eval/results_<lang>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import boto3

from signal_aid.pipeline import _pct, load_messages
from patterns.disaster_triage.questions import _REPORT_TYPES

PROMPT = """Classify this emergency message sent after an earthquake.
Return ONLY a JSON object, no other text, with these keys:
"report_type": one of {types}
"first_hand": true if the author is witnessing it in person right now, else false
"urgency": integer 0 (no risk) to 3 (a life is in danger right now)
"people_at_risk": true if it mentions people trapped, hurt or in danger, else false

Category definitions:
{defs}

Message: \"\"\"{text}\"\"\""""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/messages_sample.jsonl")
    ap.add_argument("--model-id", default=os.getenv("BEDROCK_MODEL_ID"))
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    ap.add_argument("--price-in", type=float, required=True, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, required=True, help="USD per 1M output tokens")
    a = ap.parse_args()
    if not a.model_id:
        raise SystemExit("Set --model-id or BEDROCK_MODEL_ID")

    client = boto3.client("bedrock-runtime", region_name=a.region)
    types = _REPORT_TYPES["en"]
    defs = "\n".join(f"- {k}: {v}" for k, v in types.items())
    rows = load_messages(a.data)
    lat, tin, tout, ok_type, ok_par, ok_fh, failures = [], 0, 0, 0, 0, 0, 0

    for r in rows:
        started = time.perf_counter()
        resp = client.converse(
            modelId=a.model_id,
            messages=[{"role": "user", "content": [{"text": PROMPT.format(
                types=", ".join(types), defs=defs, text=r["text"])}]}],
            inferenceConfig={"maxTokens": 120, "temperature": 0},
        )
        lat.append((time.perf_counter() - started) * 1000)
        tin += resp["usage"]["inputTokens"]
        tout += resp["usage"]["outputTokens"]
        text = resp["output"]["message"]["content"][0]["text"]
        m = re.search(r"\{.*\}", text, re.S)
        try:
            pred = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            pred = {}
        if not pred:
            failures += 1
        ok_type += pred.get("report_type") == r["label"]["report_type"]
        ok_par += bool(pred.get("people_at_risk")) == r["label"]["people_at_risk"]
        ok_fh += bool(pred.get("first_hand")) == r["label"]["first_hand"]

    n = len(rows)
    cost = tin / 1e6 * a.price_in + tout / 1e6 * a.price_out
    res = {
        "model_id": a.model_id, "n": n,
        "report_type_accuracy": round(ok_type / n, 3),
        "people_at_risk_accuracy": round(ok_par / n, 3),
        "first_hand_accuracy": round(ok_fh / n, 3),
        "unparseable_answers": failures,
        "latency_ms_p50": _pct(lat, 0.5), "latency_ms_p95": _pct(lat, 0.95),
        "input_tokens": tin, "output_tokens": tout,
        "cost_usd_total": round(cost, 6),
        "cost_usd_per_1000_messages": round(cost / n * 1000, 4),
        "note": "End-to-end latency includes the network round trip to Bedrock.",
    }
    Path("eval/baseline.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = "# Bedrock LLM baseline\n\n| Metric | Result |\n|---|---|\n" + "\n".join(
        f"| {k} | {v} |" for k, v in res.items()) + "\n"
    Path("eval/baseline.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
