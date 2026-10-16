"""Disaster Triage: the wiring that puts Decider 2B in the agent's path.

`TriageGate` is a Strands InterventionHandler. It runs before every tool call:

- publish_incident: only proceeds if the triage route is "actionable". Otherwise it
  returns Guide, so the model hands the message to a human with send_to_review.
- merge_incident: asks Decider whether the new message and the target incident are
  the same event. Below the threshold it returns Guide, so the model publishes a new
  incident instead of hiding a different emergency inside an old one.

The triage result travels in invocation_state["triage"], set by the caller:

    agent(prompt, invocation_state={"triage": {"route": route, "message": msg}})

The handler fails closed (on_error = "deny"): if Decider is down, nothing is published.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from strands.interventions import Deny, Guide, InterventionHandler, Proceed

from .questions import same_incident_question, same_incident_state
from .router import Route, Thresholds


class DeciderLike(Protocol):
    def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]: ...


# Returns (incident_summary, distance_m) for an incident id, or None if unknown.
IncidentLookup = Callable[[str, dict[str, Any]], "tuple[str, float] | None"]

GATED_TOOLS = {"publish_incident", "merge_incident"}


def check_same_incident(
    decider: DeciderLike, message_text: str, incident_summary: str, distance_m: float,
    lang: str = "en",
) -> tuple[float, float | None]:
    """Returns (P(same_incident), latency_ms)."""
    resp = decider.ask(
        same_incident_state(message_text, incident_summary, distance_m),
        same_incident_question(lang),
    )
    return float(resp["answers"]["same_incident"]["noul"]), resp.get("latency_ms")


class TriageGate(InterventionHandler):
    name = "disaster-triage-gate"

    def __init__(
        self, decider: DeciderLike, thresholds: Thresholds, lookup: IncidentLookup,
        lang: str = "en", log: list[dict[str, Any]] | None = None,
    ) -> None:
        self.decider = decider
        self.th = thresholds
        self.lookup = lookup
        self.lang = lang
        self.log = log if log is not None else []

    @property
    def on_error(self) -> str:
        return "deny"

    def before_tool_call(self, event: Any, **kwargs: Any):
        tool = event.tool_use.get("name")
        if tool not in GATED_TOOLS:
            return Proceed()

        ctx = (event.invocation_state or {}).get("triage")
        if not ctx:
            return Deny(reason="No triage result for this message. It cannot be published.")
        route: Route = ctx["route"]
        message: dict[str, Any] = ctx["message"]

        if tool == "publish_incident":
            if route.action != "actionable":
                self._record(tool, "guide", f"route is {route.action}")
                return Guide(
                    feedback=(
                        "Decider triage did not clear this message for publication "
                        f"({'; '.join(route.reasons)}). Call send_to_review instead."
                    )
                )
            self._record(tool, "proceed", "route is actionable")
            return Proceed(reason="triage: actionable")

        # merge_incident
        incident_id = str(event.tool_use.get("input", {}).get("incident_id", ""))
        found = self.lookup(incident_id, message)
        if found is None:
            self._record(tool, "deny", f"unknown incident {incident_id}")
            return Deny(reason=f"Incident {incident_id} does not exist or is not nearby.")
        summary, distance_m = found
        p, latency = check_same_incident(self.decider, message["text"], summary, distance_m, self.lang)
        if p >= self.th.same_incident_min:
            self._record(tool, "proceed", f"same_incident {p:.2f}", p, latency, incident_id)
            return Proceed(reason=f"same_incident {p:.2f}")
        self._record(tool, "guide", f"same_incident {p:.2f}", p, latency, incident_id)
        return Guide(
            feedback=(
                f"Decider says this is probably NOT the same incident as {incident_id} "
                f"(p={p:.2f}). Call publish_incident to create a new incident."
            )
        )

    def _record(self, tool: str, action: str, reason: str, p: float | None = None,
                latency_ms: float | None = None, incident_id: str | None = None) -> None:
        self.log.append({
            "tool": tool, "action": action, "reason": reason,
            "p_same_incident": p, "latency_ms": latency_ms, "incident_id": incident_id,
        })
