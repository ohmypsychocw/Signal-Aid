"""Disaster Triage: typed questions for Strands Decider 2B.

Every incoming emergency message is classified with four questions that are sent in
ONE request. Decider reads the message once and each extra question only adds its own
tokens, so asking four costs little more than asking one.

    Q1 report_type     choice  what does the message report?
    Q2 first_hand      noul    is the author witnessing it in person?
    Q3 urgency         score   0 no risk .. 3 life in danger
    Q4 people_at_risk  noul    safety net, independent of Q1

A fifth question, `same_incident`, is asked only when a nearby incident exists.

Questions are available in English ("en") and Spanish ("es"). The messages themselves
can be in any language; run eval/evaluate.py with --lang to see which works better
for your data.
"""

from __future__ import annotations

from typing import Any

from .decider import Decider

# Categories where a person may be in danger. These are NEVER auto-discarded.
CRITICAL_TYPES = {"trapped_person", "gas_leak"}

# Categories that may be handled without a human when confidence is high.
AUTO_TYPES = {"not_a_report", "offers_help"}

_REPORT_TYPES = {
    "en": {
        "trapped_person": "someone is trapped, buried or cannot get out of a place",
        "structural_damage": "a building, wall, bridge or road is damaged, and no people are mentioned in danger",
        "gas_leak": "smell of gas, a gas leak, sparks or fire risk",
        "needs_supplies": "asks for water, food, medicine, tools, shelter or volunteers",
        "offers_help": "offers supplies, transport, shelter, tools or volunteering",
        "not_a_report": "an opinion, prayer, joke, news comment or unrelated message with no situation to act on",
    },
    "es": {
        "trapped_person": "alguien está atrapado, sepultado o no puede salir de un lugar",
        "structural_damage": "un edificio, barda, puente o calle está dañado y no se mencionan personas en peligro",
        "gas_leak": "olor a gas, fuga de gas, chispas o riesgo de incendio",
        "needs_supplies": "pide agua, comida, medicinas, herramientas, refugio o voluntarios",
        "offers_help": "ofrece víveres, transporte, refugio, herramientas o ser voluntario",
        "not_a_report": "opinión, oración, broma, comentario de noticias o mensaje sin una situación que atender",
    },
}

_TEXT = {
    "en": {
        "report_type": "What does this emergency message report?",
        "first_hand": "Is the author of this message witnessing the situation in person right now?",
        "first_hand_true": "the author is there and describes what they see or hear themselves",
        "first_hand_false": "the author repeats what someone else said, shares a rumor, or is not there",
        "urgency": "How urgent is the situation described in this message?",
        "urgency_levels": [
            "no risk to anyone",
            "low risk, it can wait several hours",
            "high risk, it needs attention soon",
            "a life is in danger right now",
        ],
        "people_at_risk": "Does this message mention people who are trapped, hurt or in danger?",
        "people_true": "at least one person is described as trapped, hurt, missing or in danger",
        "people_false": "no person is described as being in danger",
        "same_incident": "Do these two messages describe the same incident at the same place?",
        "same_true": "both messages talk about the same event at the same location",
        "same_false": "they describe different events, or different places",
    },
    "es": {
        "report_type": "¿Qué reporta este mensaje de emergencia?",
        "first_hand": "¿El autor del mensaje está presenciando la situación en persona en este momento?",
        "first_hand_true": "el autor está ahí y describe lo que él mismo ve o escucha",
        "first_hand_false": "el autor repite lo que alguien más dijo, comparte un rumor o no está ahí",
        "urgency": "¿Qué tan urgente es la situación que describe este mensaje?",
        "urgency_levels": [
            "no hay riesgo para nadie",
            "riesgo bajo, puede esperar varias horas",
            "riesgo alto, necesita atención pronto",
            "hay una vida en peligro ahora mismo",
        ],
        "people_at_risk": "¿Este mensaje menciona personas atrapadas, heridas o en peligro?",
        "people_true": "al menos una persona está atrapada, herida, desaparecida o en peligro",
        "people_false": "no se describe a ninguna persona en peligro",
        "same_incident": "¿Estos dos mensajes describen el mismo incidente en el mismo lugar?",
        "same_true": "ambos mensajes hablan del mismo evento en el mismo lugar",
        "same_false": "describen eventos distintos o lugares distintos",
    },
}


def triage_questions(lang: str = "en") -> dict[str, dict[str, Any]]:
    """The four triage questions, in the wire format of POST /v1/systemone."""
    t = _TEXT[lang]
    return {
        "report_type": Decider.choice(t["report_type"], _REPORT_TYPES[lang]),
        "first_hand": Decider.noul(
            t["first_hand"], {"true": t["first_hand_true"], "false": t["first_hand_false"]}
        ),
        "urgency": Decider.score(t["urgency"], t["urgency_levels"]),
        "people_at_risk": Decider.noul(
            t["people_at_risk"], {"true": t["people_true"], "false": t["people_false"]}
        ),
    }


def same_incident_question(lang: str = "en") -> dict[str, dict[str, Any]]:
    t = _TEXT[lang]
    return {
        "same_incident": Decider.noul(
            t["same_incident"], {"true": t["same_true"], "false": t["same_false"]}
        )
    }


def same_incident_state(new_message: str, incident_summary: str, distance_m: float) -> str:
    """State for the deduplication question: the two texts plus their distance."""
    return (
        f"Message A: {new_message}\n"
        f"Message B: {incident_summary}\n"
        f"The two reports were sent {distance_m:.0f} meters apart."
    )
