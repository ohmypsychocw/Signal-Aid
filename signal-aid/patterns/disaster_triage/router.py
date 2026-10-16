"""Disaster Triage: turn Decider answers into an action, with asymmetric thresholds.

The rule that matters most: a message that may involve a person in danger is NEVER
handled automatically. It is either published (high confidence) or sent to a human
as a priority. Only clearly harmless messages are auto-handled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .questions import AUTO_TYPES, CRITICAL_TYPES

DEFAULT_THRESHOLDS = Path(__file__).with_name("thresholds.yaml")


@dataclass
class Thresholds:
    publish_min_conf: float = 0.9
    auto_min_conf: float = 0.9
    people_at_risk_flag: float = 0.3
    first_hand_min: float = 0.5
    same_incident_min: float = 0.8
    dedupe_radius_m: float = 200.0
    dedupe_window_min: float = 120.0

    @classmethod
    def load(cls, path: str | Path = DEFAULT_THRESHOLDS) -> "Thresholds":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**{k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Triage:
    report_type: str
    report_conf: float
    report_probs: dict[str, float]
    first_hand: float
    urgency: float
    urgency_conf: float
    people_at_risk: float
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Route:
    action: str  # "actionable" | "human_review" | "auto_handle"
    priority: bool
    first_hand: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_triage(response: dict[str, Any]) -> Triage:
    """Parse a /v1/systemone response that answered triage_questions()."""
    a = response["answers"]
    rt = a["report_type"]
    ug = a["urgency"]
    return Triage(
        report_type=rt["choice"],
        report_conf=float(rt["confidence"]),
        report_probs={k: float(v) for k, v in rt["probabilities"].items()},
        first_hand=float(a["first_hand"]["noul"]),
        urgency=float(ug["score"]),
        urgency_conf=float(ug["confidence"]),
        people_at_risk=float(a["people_at_risk"]["noul"]),
        latency_ms=response.get("latency_ms"),
    )


def route(t: Triage, th: Thresholds) -> Route:
    first_hand = t.first_hand >= th.first_hand_min
    safety = t.report_type in CRITICAL_TYPES or t.people_at_risk >= th.people_at_risk_flag

    if safety:
        reasons = []
        if t.report_type in CRITICAL_TYPES:
            reasons.append(f"critical category {t.report_type}")
        if t.people_at_risk >= th.people_at_risk_flag:
            reasons.append(f"people_at_risk {t.people_at_risk:.2f} ≥ {th.people_at_risk_flag}")
        confident = t.report_conf >= th.publish_min_conf and t.report_type in CRITICAL_TYPES
        if confident:
            reasons.append(f"confidence {t.report_conf:.2f} ≥ {th.publish_min_conf}")
            return Route("actionable", True, first_hand, reasons)
        reasons.append("possible person in danger: a human must decide")
        return Route("human_review", True, first_hand, reasons)

    if t.report_type in AUTO_TYPES and t.report_conf >= th.auto_min_conf:
        return Route(
            "auto_handle", False, first_hand,
            [f"{t.report_type} with confidence {t.report_conf:.2f} ≥ {th.auto_min_conf}"],
        )

    if t.report_conf >= th.publish_min_conf and t.report_type not in AUTO_TYPES:
        return Route(
            "actionable", t.urgency >= 2.0, first_hand,
            [f"{t.report_type} with confidence {t.report_conf:.2f}"],
        )

    return Route(
        "human_review", False, first_hand,
        [f"low confidence ({t.report_conf:.2f}) for {t.report_type}"],
    )
