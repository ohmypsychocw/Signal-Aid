"""Offline tests for the Disaster Triage pattern (no GPU, no Bedrock)."""

from types import SimpleNamespace

from strands.interventions import Deny, Guide, Proceed

from patterns.disaster_triage import Thresholds, Triage, route
from patterns.disaster_triage.handler import TriageGate

TH = Thresholds()


def tri(rt, conf, people=0.0, fh=0.9, urg=1.0):
    return Triage(rt, conf, {rt: conf}, fh, urg, 0.8, people)


def test_person_at_risk_is_never_auto_handled():
    # Even if Q1 says "not_a_report" with high confidence, Q4 keeps it for a human.
    r = route(tri("not_a_report", 0.99, people=0.35), TH)
    assert r.action == "human_review" and r.priority


def test_confident_trapped_person_is_published_as_priority():
    r = route(tri("trapped_person", 0.95, people=0.9, urg=3), TH)
    assert r.action == "actionable" and r.priority


def test_unsure_trapped_person_goes_to_priority_review():
    r = route(tri("trapped_person", 0.6, people=0.9), TH)
    assert r.action == "human_review" and r.priority


def test_harmless_message_is_auto_handled():
    assert route(tri("offers_help", 0.95), TH).action == "auto_handle"


def test_low_confidence_goes_to_review():
    assert route(tri("structural_damage", 0.5), TH).action == "human_review"


class FakeDecider:
    def __init__(self, p):
        self.p = p

    def ask(self, state, questions):
        return {"answers": {"same_incident": {"type": "noul", "noul": self.p}}, "latency_ms": 1.0}


def event(tool, inp, rt):
    return SimpleNamespace(
        tool_use={"name": tool, "input": inp, "toolUseId": "t1"},
        invocation_state={"triage": {"route": rt, "message": {"id": "M1", "text": "x"}}},
    )


def test_gate_guides_publish_when_not_actionable():
    gate = TriageGate(FakeDecider(0.9), TH, lambda i, m: None)
    rt = route(tri("trapped_person", 0.5, people=0.9), TH)
    assert isinstance(gate.before_tool_call(event("publish_incident", {}, rt)), Guide)


def test_gate_merges_only_when_decider_agrees():
    rt = route(tri("trapped_person", 0.95, people=0.9), TH)
    lookup = lambda i, m: ("trapped_person at Calle Morelos 210", 40.0) if i == "INC-001" else None
    assert isinstance(TriageGate(FakeDecider(0.92), TH, lookup).before_tool_call(
        event("merge_incident", {"incident_id": "INC-001"}, rt)), Proceed)
    assert isinstance(TriageGate(FakeDecider(0.40), TH, lookup).before_tool_call(
        event("merge_incident", {"incident_id": "INC-001"}, rt)), Guide)
    assert isinstance(TriageGate(FakeDecider(0.92), TH, lookup).before_tool_call(
        event("merge_incident", {"incident_id": "INC-999"}, rt)), Deny)


def test_gate_ignores_other_tools():
    rt = route(tri("offers_help", 0.95), TH)
    assert isinstance(TriageGate(FakeDecider(0.5), TH, lambda i, m: None).before_tool_call(
        event("send_to_review", {}, rt)), Proceed)
