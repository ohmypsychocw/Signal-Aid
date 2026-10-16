"""Generate synthetic post-earthquake messages in colloquial Mexican Spanish with Amazon Bedrock.

    export BEDROCK_MODEL_ID=<model id or inference profile>
    python data/generate_synthetic.py --n 300 --out data/generated.jsonl

The generated labels are a FIRST DRAFT. Review every label by hand before using the
file for evaluation. All messages are fictional: no real people, no real events.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

import boto3

SITES = [
    ("Calle Morelos 210", 20.67652, -103.34805),
    ("Av. Juárez 455", 20.67401, -103.35302),
    ("Calle Pino Suárez 88", 20.67903, -103.34411),
    ("Calle Libertad 1020", 20.67198, -103.36004),
    ("Calle Hidalgo 300", 20.67788, -103.35102),
    ("Calle Colón 150", 20.67309, -103.34650),
]
TYPES = ["trapped_person", "structural_damage", "gas_leak", "needs_supplies", "offers_help", "not_a_report"]

PROMPT = """Write {k} short, realistic messages that people in a Mexican city might post in
the first hour after a FICTIONAL strong earthquake. Write them in colloquial Mexican Spanish,
the way people really type in group chats: typos, no accents sometimes, slang, emojis,
capital letters when they panic. Mix these situations: {types}.
Include some rumors written as if second-hand ("dicen que...", "me pasaron que...") and a few
tricky ones (a rumor written in first person, a harmless message that sounds scary).
When a message names a place, use one of these addresses: {sites}.

Return ONLY a JSON array. Each item:
{{"text": "...", "address": "<one of the addresses or null>",
  "label": {{"report_type": "<one of {types}>", "people_at_risk": true|false, "first_hand": true|false}}}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out", default="data/generated.jsonl")
    ap.add_argument("--model-id", default=os.getenv("BEDROCK_MODEL_ID"))
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    ap.add_argument("--seed", type=int, default=19)
    a = ap.parse_args()
    if not a.model_id:
        raise SystemExit("Set --model-id or BEDROCK_MODEL_ID")

    random.seed(a.seed)
    client = boto3.client("bedrock-runtime", region_name=a.region)
    sites = {s[0]: s for s in SITES}
    out, minute = [], 0
    while len(out) < a.n:
        prompt = PROMPT.format(k=a.batch, types=", ".join(TYPES), sites="; ".join(sites))
        resp = client.converse(
            modelId=a.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 4000, "temperature": 0.9},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            continue
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for it in items:
            if it.get("label", {}).get("report_type") not in TYPES:
                continue
            site = sites.get(it.get("address") or "")
            if site:
                lat, lon = site[1] + random.uniform(-2e-4, 2e-4), site[2] + random.uniform(-2e-4, 2e-4)
            else:
                lat, lon = 20.6755 + random.uniform(-4e-3, 4e-3), -103.35 + random.uniform(-4e-3, 4e-3)
            minute += random.choice([0, 0, 1])
            h, mm = divmod(21 + minute, 60)
            out.append({
                "id": f"GEN-{len(out) + 1:04d}", "ts": f"2026-11-05T{13 + h:02d}:{mm:02d}:00-06:00",
                "text": it["text"], "lat": round(lat, 5), "lon": round(lon, 5),
                "address": site[0] if site else None, "label": it["label"],
            })
        print(f"{len(out)} messages")

    with open(a.out, "w", encoding="utf-8") as f:
        for row in out[: a.n]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {min(len(out), a.n)} messages to {a.out}. Review the labels by hand.")


if __name__ == "__main__":
    main()
