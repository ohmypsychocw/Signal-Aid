"""Run Signal Aid over a stream of messages and write a replay file for the web demo.

    # Real run (Decider server on localhost + Amazon Bedrock):
    python -m signal_aid.pipeline --messages data/messages_sample.jsonl --out web/replay.json

    # Without Bedrock (deterministic briefs, Decider still decides):
    python -m signal_aid.pipeline --no-llm

    # UI development only, no GPU and no Bedrock. The replay is labeled MOCK:
    python -m signal_aid.pipeline --mock-decider --no-llm
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from patterns.disaster_triage import Thresholds, parse_triage, route, triage_questions
from patterns.disaster_triage.handler import TriageGate, check_same_incident

from patterns.disaster_triage.decider import DEFAULT_URL

from .decider_client import DeciderClient, MockDecider
from .store import IncidentStore

TEMPLATE_BRIEF = {
    "trapped_person": "Posible persona atrapada",
    "structural_damage": "Daño estructural reportado",
    "gas_leak": "Posible fuga de gas",
    "needs_supplies": "Se solicitan insumos",
}


def load_messages(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["ts"])


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1)


def _no_llm_dispatch(msg, triage, rt, store, decider, th, lang, gate_log):
    """Same decisions as the agent, without Bedrock. Briefs come from a template."""
    for inc, d in store.nearby(msg, th.dedupe_radius_m, th.dedupe_window_min):
        p, lat = check_same_incident(decider, msg["text"], f"{inc.category} at {inc.address}: {inc.brief}", d, lang)
        ok = p >= th.same_incident_min
        gate_log.append({"tool": "merge_incident", "action": "proceed" if ok else "guide",
                         "reason": f"same_incident {p:.2f}", "p_same_incident": p,
                         "latency_ms": lat, "incident_id": inc.id})
        if ok:
            n = len(inc.report_ids) + 1
            brief = f"{TEMPLATE_BRIEF.get(inc.category, inc.category)} · {n} reportes"
            store.merge(inc.id, msg, brief, rt.first_hand, rt.priority)
            return {"result": "merged", "incident_id": inc.id}
    cat = triage.report_type
    inc = store.publish(msg, cat, msg.get("address") or "sin dirección",
                        f"{TEMPLATE_BRIEF.get(cat, cat)} · 1 reporte", rt.priority, rt.first_hand)
    gate_log.append({"tool": "publish_incident", "action": "proceed", "reason": "route is actionable"})
    return {"result": "published", "incident_id": inc.id}


def run(messages: list[dict[str, Any]], decider, th: Thresholds, use_llm: bool, lang: str) -> dict[str, Any]:
    store = IncidentStore()
    gate = TriageGate(decider, th, store.lookup_for(th.dedupe_radius_m, th.dedupe_window_min), lang)
    model = None
    if use_llm:
        from .agent import make_model
        model = make_model()

    questions = triage_questions(lang)
    events, triage_lat, triage_rt, bedrock_calls = [], [], [], 0
    started = time.perf_counter()

    for msg in messages:
        resp = decider.ask(msg["text"], questions)
        triage = parse_triage(resp)
        if resp.get("latency_ms"):
            triage_lat.append(float(resp["latency_ms"]))
        if resp.get("roundtrip_ms"):
            triage_rt.append(float(resp["roundtrip_ms"]))
        rt = route(triage, th)
        gate.log = []

        if rt.action == "auto_handle":
            if triage.report_type == "offers_help":
                store.volunteer_offers.append({"message_id": msg["id"], "text": msg["text"]})
            outcome = {"result": "auto_handled"}
        elif rt.action == "human_review":
            store.review_queue.append({"message_id": msg["id"], "reason": "; ".join(rt.reasons),
                                       "priority": rt.priority})
            outcome = {"result": "human_review"}
        elif use_llm:
            from .agent import dispatch
            bedrock_calls += 1
            outcome = dispatch(msg, triage, rt, store, gate, model)
        else:
            outcome = _no_llm_dispatch(msg, triage, rt, store, decider, th, lang, gate.log)

        events.append({
            "message": msg, "triage": triage.to_dict(), "route": rt.to_dict(),
            "outcome": outcome, "gate": list(gate.log),
        })

    elapsed = time.perf_counter() - started
    counts: dict[str, int] = {}
    for e in events:
        counts[e["outcome"]["result"]] = counts.get(e["outcome"]["result"], 0) + 1

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mock_decider": bool(getattr(decider, "is_mock", False)),
            "decider_model": getattr(decider, "model", "unknown"),
            "llm": use_llm,
            "question_lang": lang,
            "thresholds": th.__dict__,
            "synthetic_data": True,
        },
        "stats": {
            "messages": len(events),
            "outcomes": counts,
            "incidents": len(store.incidents),
            "human_review": len(store.review_queue),
            "bedrock_agent_invocations": bedrock_calls,
            "bedrock_invocations_avoided": len(events) - bedrock_calls if use_llm else None,
            "decider_triage_latency_ms_p50": _pct(triage_lat, 0.5),
            "decider_triage_latency_ms_p95": _pct(triage_lat, 0.95),
            "decider_roundtrip_ms_p50": _pct(triage_rt, 0.5),
            "wall_clock_s": round(elapsed, 2),
        },
        "events": events,
        "incidents": [i.to_dict() for i in store.incidents.values()],
        "review_queue": store.review_queue,
        "volunteer_offers": store.volunteer_offers,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--messages", default="data/messages_sample.jsonl")
    ap.add_argument("--out", default="web/replay.json")
    ap.add_argument("--decider-url", default=DEFAULT_URL)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--lang", default="en", choices=["en", "es"], help="language of the questions")
    ap.add_argument("--no-llm", action="store_true", help="skip Amazon Bedrock")
    ap.add_argument("--mock-decider", action="store_true", help="UI development only")
    a = ap.parse_args()

    th = Thresholds.load(a.thresholds) if a.thresholds else Thresholds.load()
    decider = MockDecider() if a.mock_decider else DeciderClient(a.decider_url)
    print("Decider:", decider.health())
    replay = run(load_messages(a.messages), decider, th, use_llm=not a.no_llm, lang=a.lang)
    Path(a.out).write_text(json.dumps(replay, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(replay["stats"], indent=2))
    print(f"Replay written to {a.out}")


if __name__ == "__main__":
    main()
