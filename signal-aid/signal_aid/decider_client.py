"""Decider clients used by Signal Aid.

`DeciderClient` is the pattern's standard-library client (patterns/disaster_triage/decider.py).

MockDecider exists ONLY to develop the web UI without a GPU. It uses keyword rules,
is not a model, and every replay it produces is labeled "mock" in the UI.
Never report MockDecider numbers as results.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any

from patterns.disaster_triage.decider import Decider as DeciderClient  # noqa: F401


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


_KEYWORDS = {
    "trapped_person": ["atrapad", "gritos", "no puede salir", "bajo los escombros", "sepultad", "debajo de"],
    "gas_leak": ["gas", "chispas", "humo", "incendio"],
    "structural_damage": ["se cayo", "grieta", "colaps", "barda", "derrumb", "puente", "cuarteado"],
    "needs_supplies": ["necesit", "faltan", "urge agua", "medicinas", "palas", "cascos"],
    "offers_help": ["ofrezco", "llevo", "tengo camioneta", "puedo ayudar", "voluntario", "donamos"],
}
_PEOPLE = ["atrapad", "herid", "gritos", "no contesta", "desaparec", "nino", "abuel", "persona", "senora", "senor"]
_SECOND_HAND = ["dicen", "me dijeron", "escuche que", "rt ", "alguien sabe", "vi en face", "segun"]


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values())
    e = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(e.values())
    return {k: round(v / z, 4) for k, v in e.items()}


class MockDecider:
    """Keyword heuristic with the same response shape as the real server. UI dev only."""

    is_mock = True
    model = "MOCK-keyword-rules"

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.model, "mock": True}

    def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        text = _norm(state)
        answers: dict[str, Any] = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                scores = {k: 0.0 for k in q["criteria"]}
                for cat, kws in _KEYWORDS.items():
                    if cat in scores:
                        scores[cat] += 4.5 * sum(kw in text for kw in kws)
                if "not_a_report" in scores and max(scores.values()) == 0:
                    scores["not_a_report"] = 4.0
                probs = _softmax(scores)
                best = max(probs, key=probs.get)
                answers[name] = {"type": "choice", "choice": best, "probabilities": probs,
                                 "confidence": probs[best]}
            elif q["type"] == "score":
                levels = q["criteria"]
                lvl = 0
                if any(k in text for k in _KEYWORDS["structural_damage"] + _KEYWORDS["needs_supplies"]):
                    lvl = 2
                if any(k in text for k in _KEYWORDS["trapped_person"] + _KEYWORDS["gas_leak"]):
                    lvl = 3
                probs = {str(i): (0.7 if i == lvl else 0.3 / (len(levels) - 1)) for i in range(len(levels))}
                answers[name] = {"type": "score", "score": float(lvl),
                                 "legend": {str(i): l for i, l in enumerate(levels)},
                                 "probabilities": probs, "confidence": 0.7}
            else:
                if name == "people_at_risk":
                    p = 0.92 if any(k in text for k in _PEOPLE) else 0.08
                elif name == "first_hand":
                    p = 0.2 if any(k in text for k in _SECOND_HAND) else 0.8
                elif name == "same_incident":
                    a, _, b = text.partition("message b:")
                    tok = lambda x: {w.strip(".,:!?") for w in x.split() if len(w) > 4 or w.strip(".,:!?").isdigit()}
                    shared = tok(a) & tok(b)
                    p = min(0.95, 0.15 + 0.2 * len(shared) + (0.45 if any(w.isdigit() for w in shared) else 0))
                else:
                    p = 0.5
                answers[name] = {"type": "noul", "noul": round(p, 4)}
        return {"model": self.model, "answers": answers, "usage": {}, "latency_ms": 0.0, "mock": True}
