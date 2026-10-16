"""Signal Aid in one screen: four messages, four decisions, printed as they happen.

Mirrors the official strands-decider example (examples/strands/tool_call_intervention.py):
Decider classifies, plain Python decides, and a Strands InterventionHandler gates the
tool calls of a Bedrock agent.

    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
    python examples/quickstart.py            # needs AWS credentials with Bedrock access
    python examples/quickstart.py --no-llm   # Decider only

STRANDS_DECIDER_URL overrides the endpoint; BEDROCK_MODEL_ID overrides the Bedrock model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patterns.disaster_triage import Decider, ServerUnavailable, Thresholds, parse_triage, route, triage_questions  # noqa: E402
from patterns.disaster_triage.handler import TriageGate  # noqa: E402
from signal_aid.store import IncidentStore  # noqa: E402

MESSAGES = [
    # A first-hand report of a trapped person: should be published as a priority incident.
    {"id": "Q-1", "ts": "2026-11-05T13:21:00-06:00", "lat": 20.67652, "lon": -103.34805,
     "address": "Calle Morelos 210",
     "text": "ayuda!!! en calle morelos 210 se cayó la casa y escucho gritos debajo, somos vecinos no podemos solos"},
    # The same incident, three minutes later: should be merged, not published twice.
    {"id": "Q-2", "ts": "2026-11-05T13:24:00-06:00", "lat": 20.67660, "lon": -103.34800,
     "address": "Calle Morelos 210",
     "text": "Morelos 210 sigue la señora abajo, ya la oímos pidiendo agua, urgen brigadistas con herramienta"},
    # Harmless: should be handled automatically.
    {"id": "Q-3", "ts": "2026-11-05T13:25:00-06:00", "lat": 20.6750, "lon": -103.3500, "address": None,
     "text": "que fuerte estuvo, dios cuide a todos 🙏"},
    # Sounds casual, but a person is hurt: the safety net must keep it away from auto-handling.
    {"id": "Q-4", "ts": "2026-11-05T13:26:00-06:00", "lat": 20.6738, "lon": -103.3462, "address": None,
     "text": "jaja no manches mi tía dice que ya no siente las piernas desde que se le cayó el ropero encima"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip Amazon Bedrock")
    ap.add_argument("--lang", default="en", choices=["en", "es"])
    a = ap.parse_args()

    decider = Decider()
    try:
        h = decider.health()
        print(f"strands-decider: {h.get('checkpoint')} on {h.get('device')} at {decider.url}\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    th = Thresholds.load()
    store = IncidentStore()
    gate = TriageGate(decider, th, store.lookup_for(th.dedupe_radius_m, th.dedupe_window_min), a.lang)
    model = None
    if not a.no_llm:
        from signal_aid.agent import dispatch, make_model
        model = make_model()

    for msg in MESSAGES:
        resp = decider.ask(msg["text"], triage_questions(a.lang))
        t = parse_triage(resp)
        rt = route(t, th)
        print(f"{msg['id']}: “{msg['text']}”")
        print(f"    report_type    : {t.report_type} ({t.report_conf:.2f})")
        print(f"    first_hand     : {t.first_hand:.2f}")
        print(f"    urgency        : {t.urgency:.1f} / 3")
        print(f"    people_at_risk : {t.people_at_risk:.2f}")
        print(f"    decided in     : {resp.get('latency_ms')} ms, four questions in one call")
        print(f"    -> {rt.action}{' (priority)' if rt.priority else ''}: {'; '.join(rt.reasons)}")

        if rt.action == "actionable":
            gate.log = []
            if model is not None:
                out = dispatch(msg, t, rt, store, gate, model)
            else:
                from signal_aid.pipeline import _no_llm_dispatch
                out = _no_llm_dispatch(msg, t, rt, store, decider, th, a.lang, gate.log)
            for g in gate.log:
                print(f"    gate: {g['tool']} ~> {g['action']} ({g['reason']})")
            print(f"    outcome: {out.get('result')} {out.get('incident_id') or ''}")
        print()

    print("Incidents on the map:")
    for inc in store.incidents.values():
        d = inc.to_dict()
        print(f"  {d['id']} · {d['address']} · {d['category']} · {len(d['report_ids'])} report(s) · {d['status']}")
        print(f"      {d['brief']}")
    print(f"\n{decider.calls} Decider calls in total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
