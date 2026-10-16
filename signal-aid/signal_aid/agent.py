"""The Signal Aid dispatcher agent: Amazon Bedrock writes, Decider 2B gates.

Bedrock is only invoked for messages Decider routed as "actionable" (or for the
deduplication of those). Every publish/merge tool call passes through TriageGate.
"""

from __future__ import annotations

import os
from typing import Any

from strands import Agent, tool
from strands.models import BedrockModel

from patterns.disaster_triage import Route, Thresholds, Triage
from patterns.disaster_triage.handler import TriageGate

from .store import IncidentStore

SYSTEM_PROMPT = """You are the dispatcher of Signal Aid, a support tool for volunteer rescue
brigades after an earthquake in Mexico. You receive ONE emergency message that a
decision model already classified, plus a list of nearby open incidents.

Do exactly one of these, by calling exactly one tool:
- merge_incident, if a nearby incident clearly describes the same event at the same place;
- publish_incident, if this is a new incident;
- send_to_review, if you are told to, or if the message is too unclear to act on.

Rules:
- address: copy the street and number exactly as written in the message. Never invent one.
  If the message has none, use "sin dirección".
- brief: one sentence in Spanish, at most 25 words, for a brigade leader. Say what
  happened, whether people are at risk and how many reports confirm it. No speculation.
- If a tool call is refused with feedback, follow the feedback.
After the tool call, answer with one word: done."""


# The same Bedrock model the official Strands Decider example uses. Override with BEDROCK_MODEL_ID.
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def make_model() -> BedrockModel:
    kwargs: dict[str, Any] = {"temperature": 0.1,
                              "model_id": os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)}
    if os.getenv("AWS_REGION"):
        kwargs["region_name"] = os.environ["AWS_REGION"]
    return BedrockModel(**kwargs)


def dispatch(msg: dict[str, Any], triage: Triage, rt: Route, store: IncidentStore,
             gate: TriageGate, model: BedrockModel | None = None) -> dict[str, Any]:
    """Run the agent for one message. Returns what happened."""
    outcome: dict[str, Any] = {"result": "none"}

    @tool
    def publish_incident(category: str, address: str, brief: str) -> str:
        """Create a new incident on the brigades' map.

        Args:
            category: one of trapped_person, structural_damage, gas_leak, needs_supplies.
            address: street and number exactly as written in the message, or "sin dirección".
            brief: one sentence in Spanish, at most 25 words.
        """
        inc = store.publish(msg, category, address, brief, rt.priority, rt.first_hand)
        outcome.update(result="published", incident_id=inc.id)
        return f"Published {inc.id}."

    @tool
    def merge_incident(incident_id: str, brief: str) -> str:
        """Add this message as a new report of an existing nearby incident.

        Args:
            incident_id: id of the nearby incident, for example INC-004.
            brief: updated one-sentence brief in Spanish that counts all reports.
        """
        inc = store.merge(incident_id, msg, brief, rt.first_hand, rt.priority)
        outcome.update(result="merged", incident_id=inc.id)
        return f"Merged into {inc.id}."

    @tool
    def send_to_review(reason: str) -> str:
        """Send this message to a human reviewer instead of publishing it.

        Args:
            reason: short reason in English.
        """
        store.review_queue.append({"message_id": msg["id"], "reason": reason, "priority": rt.priority})
        outcome.update(result="human_review", reason=reason)
        return "Sent to human review."

    th: Thresholds = gate.th
    nearby = store.nearby(msg, th.dedupe_radius_m, th.dedupe_window_min)
    nearby_txt = "\n".join(
        f"- {inc.id} ({d:.0f} m away): {inc.category} at {inc.address}. {inc.brief}"
        for inc, d in nearby
    ) or "- none"

    prompt = (
        f"Message {msg['id']} sent at {msg['ts']}:\n\"{msg['text']}\"\n\n"
        f"Decider triage: report_type={triage.report_type} (confidence {triage.report_conf:.2f}), "
        f"people_at_risk={triage.people_at_risk:.2f}, first_hand={triage.first_hand:.2f}, "
        f"urgency={triage.urgency:.1f}/3. Route: {rt.action}.\n\n"
        f"Nearby open incidents:\n{nearby_txt}"
    )

    agent = Agent(
        model=model or make_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[publish_incident, merge_incident, send_to_review],
        interventions=[gate],
        callback_handler=None,
    )
    result = agent(prompt, invocation_state={"triage": {"route": rt, "message": msg}})

    usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None) or {}
    outcome["bedrock_usage"] = dict(usage) if isinstance(usage, dict) else {}
    outcome["nearby"] = [inc.id for inc, _ in nearby]
    return outcome
