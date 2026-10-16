"""Evaluate the Disaster Triage pattern on a labeled set.

    python -m eval.evaluate --data data/messages_sample.jsonl --lang en
    python -m eval.evaluate --data data/messages_sample.jsonl --lang es

Each line needs: text, label.report_type, label.people_at_risk, label.first_hand.
Writes eval/results_<lang>.md and eval/results_<lang>.json.

The metric that matters most is SAFETY RECALL: of the messages where a person is
in danger, how many were NOT auto-handled (they reached the map or a human).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patterns.disaster_triage.decider import DEFAULT_URL
from signal_aid.decider_client import DeciderClient, MockDecider
from signal_aid.pipeline import _pct, load_messages
from patterns.disaster_triage import Thresholds, parse_triage, route, triage_questions


def evaluate(rows, decider, th: Thresholds, lang: str) -> dict:
    qs = triage_questions(lang)
    preds = []
    for r in rows:
        resp = decider.ask(r["text"], qs)
        t = parse_triage(resp)
        preds.append((r, t, route(t, th), resp.get("latency_ms")))

    n = len(preds)
    acc_type = sum(t.report_type == r["label"]["report_type"] for r, t, _, _ in preds) / n
    acc_fh = sum((t.first_hand >= 0.5) == r["label"]["first_hand"] for r, t, _, _ in preds) / n
    acc_par = sum((t.people_at_risk >= 0.5) == r["label"]["people_at_risk"] for r, t, _, _ in preds) / n

    at_risk = [(r, t, rt) for r, t, rt, _ in preds if r["label"]["people_at_risk"]]
    missed = [r["id"] for r, t, rt in at_risk if rt.action == "auto_handle"]
    safety_recall = (len(at_risk) - len(missed)) / len(at_risk) if at_risk else None

    auto = [(r, rt) for r, t, rt, _ in preds if rt.action == "auto_handle"]
    auto_ok = sum(r["label"]["report_type"] in {"not_a_report", "offers_help"}
                  and not r["label"]["people_at_risk"] for r, _ in auto)

    bins = []
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        sel = [(r, t) for r, t, _, _ in preds if lo <= t.report_conf < hi]
        if sel:
            bins.append({"range": f"{lo:.1f}–{min(hi, 1.0):.1f}", "n": len(sel),
                         "accuracy": round(sum(t.report_type == r["label"]["report_type"] for r, t in sel) / len(sel), 3)})

    lat = [float(l) for *_, l in preds if l]
    actions: dict[str, int] = {}
    for _, _, rt, _ in preds:
        actions[rt.action] = actions.get(rt.action, 0) + 1

    errors = [{"id": r["id"], "text": r["text"], "label": r["label"]["report_type"],
               "pred": t.report_type, "conf": round(t.report_conf, 3), "route": rt.action}
              for r, t, rt, _ in preds if t.report_type != r["label"]["report_type"]]

    return {
        "n": n, "lang": lang, "mock_decider": bool(getattr(decider, "is_mock", False)),
        "report_type_accuracy": round(acc_type, 3),
        "first_hand_accuracy": round(acc_fh, 3),
        "people_at_risk_accuracy": round(acc_par, 3),
        "safety_recall": None if safety_recall is None else round(safety_recall, 3),
        "missed_people_at_risk": missed,
        "routes": actions,
        "auto_handled_precision": round(auto_ok / len(auto), 3) if auto else None,
        "calibration": bins,
        "latency_ms_p50": _pct(lat, 0.5), "latency_ms_p95": _pct(lat, 0.95),
        "errors": errors,
    }


def to_markdown(res: dict) -> str:
    warn = "\n> ⚠️ MOCK decider: these numbers are NOT Decider 2B results.\n" if res["mock_decider"] else ""
    lines = [
        f"# Disaster Triage evaluation (questions in `{res['lang']}`, n = {res['n']})", warn,
        "| Metric | Result |", "|---|---|",
        f"| Report type accuracy | {res['report_type_accuracy']:.1%} |",
        f"| First-hand accuracy | {res['first_hand_accuracy']:.1%} |",
        f"| People-at-risk accuracy | {res['people_at_risk_accuracy']:.1%} |",
        f"| **Safety recall** (people at risk not auto-handled) | {res['safety_recall']:.1%} |"
        if res["safety_recall"] is not None else "| Safety recall | n/a |",
        f"| Auto-handled precision | {res['auto_handled_precision']:.1%} |"
        if res["auto_handled_precision"] is not None else "| Auto-handled precision | n/a |",
        f"| Decider latency p50 / p95 | {res['latency_ms_p50']} ms / {res['latency_ms_p95']} ms |",
        "", "## Routes", "", *[f"- {k}: {v}" for k, v in res["routes"].items()],
        "", "## Calibration of report_type", "", "| Confidence | n | Accuracy |", "|---|---|---|",
        *[f"| {b['range']} | {b['n']} | {b['accuracy']:.1%} |" for b in res["calibration"]],
        "", "## Misclassified messages", "", "| id | label | predicted | conf | route | text |",
        "|---|---|---|---|---|---|",
        *[f"| {e['id']} | {e['label']} | {e['pred']} | {e['conf']} | {e['route']} | {e['text'][:70]} |"
          for e in res["errors"]],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/messages_sample.jsonl")
    ap.add_argument("--lang", default="en", choices=["en", "es"])
    ap.add_argument("--decider-url", default=DEFAULT_URL)
    ap.add_argument("--mock-decider", action="store_true")
    a = ap.parse_args()

    decider = MockDecider() if a.mock_decider else DeciderClient(a.decider_url)
    res = evaluate(load_messages(a.data), decider, Thresholds.load(), a.lang)
    out = Path("eval")
    (out / f"results_{a.lang}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(res)
    (out / f"results_{a.lang}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
