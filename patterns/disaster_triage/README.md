# Reusable decision pattern: Disaster Triage (`#policy`)

Classify every incoming emergency message **before** the agent acts on it, with four
typed Strands Decider 2B questions in one call, and route it with **asymmetric
thresholds** so a person in danger is never handled automatically.

| Part | File |
|---|---|
| Standard-library Decider client with `noul` / `choice` / `score` builders | `decider.py` |
| Typed questions and option sets (English and Spanish) | `questions.py` |
| Parsing answers and asymmetric routing | `router.py` |
| Wiring into the agent: a Strands `InterventionHandler` | `handler.py` |
| Thresholds, with the reason for each one | `thresholds.yaml` |

## The questions

| Name | Type | Question | Options |
|---|---|---|---|
| `report_type` | choice | What does this emergency message report? | `trapped_person`, `structural_damage`, `gas_leak`, `needs_supplies`, `offers_help`, `not_a_report` |
| `first_hand` | noul | Is the author witnessing the situation in person right now? | true / false, with descriptions |
| `urgency` | score | How urgent is the situation? | 4 ordered levels, from "no risk" to "a life is in danger right now" |
| `people_at_risk` | noul | Does it mention people trapped, hurt or in danger? | true / false (**safety net**) |
| `same_incident` | noul | Do these two messages describe the same incident? | asked only for nearby incidents |

Every option has a one-line description. In our tests the descriptions changed results
more than the option names.

## The routing rule

| Condition | Action |
|---|---|
| `report_type` is critical (`trapped_person`, `gas_leak`) or `people_at_risk ≥ 0.3` | Never auto-handled. Published if confident, otherwise **priority human review** |
| `not_a_report` or `offers_help` with confidence ≥ 0.9 | Handled automatically |
| Any other category with confidence ≥ 0.9 | Actionable |
| Everything else | Human review |

`people_at_risk` is independent of `report_type`: if Q1 is wrong, Q4 still catches it.

## Use it in your agent

Start a server first (no authentication, keep it on localhost):

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
```

```python
from strands import Agent
from patterns.disaster_triage import Decider, Thresholds, parse_triage, route, triage_questions
from patterns.disaster_triage.handler import TriageGate

decider = Decider()          # http://127.0.0.1:8099, or STRANDS_DECIDER_URL
th = Thresholds.load()
gate = TriageGate(decider, th, lookup=my_incident_lookup)
agent = Agent(tools=[publish_incident, merge_incident, send_to_review], interventions=[gate])

triage = parse_triage(decider.ask(message["text"], triage_questions("en")))
rt = route(triage, th)
if rt.action == "actionable":
    agent(prompt, invocation_state={"triage": {"route": rt, "message": message}})
```

`TriageGate.before_tool_call` returns:

- `Proceed` when the triage allows the call, or Decider confirms a duplicate;
- `Guide` when the model should do something else (send to review, or publish instead of merging);
- `Deny` when the target incident does not exist, or there is no triage result.

It fails closed (`on_error = "deny"`): if Decider is unreachable, nothing is published.

## Adapt it in 3 steps

1. **Change the categories** in `_REPORT_TYPES` (for floods: `flooded_home`, `stranded_on_roof`, `road_blocked`…) and update `CRITICAL_TYPES` and `AUTO_TYPES`.
2. **Keep a safety-net question** like `people_at_risk` that does not depend on the main category.
3. **Tune `thresholds.yaml`** with `python -m eval.evaluate` on a small labeled set of your own, and check that safety recall stays at 100%.

## Limits

Decider 2B is experimental and adversarial text can sway it. Use this pattern to route
and prioritize, never as the only safeguard, and keep humans in the loop.
