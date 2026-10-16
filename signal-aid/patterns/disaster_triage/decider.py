"""A small synchronous client for a running Strands Decider server.

Follows the shape of the official example client in strands-labs/strands-decider
(examples/strands/_client.py, Apache-2.0): standard library only, so the pattern can be
dropped into any Strands project that already has `strands-agents`, and one request at a
time, which is how the server is best driven.

Start a server first (it has no authentication; keep it on localhost):

    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099

STRANDS_DECIDER_URL overrides the endpoint (default http://127.0.0.1:8099).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

DEFAULT_URL = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")
SERVE_HINT = "strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099"


class ServerUnavailable(RuntimeError):
    """The Decider server is not answering."""


class Decider:
    """One state, many typed questions, one round trip.

    The state is encoded once and shared across the questions, so asking four things
    costs far less than four requests. Ask everything you need at once.
    """

    is_mock = False

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0, model: str = "strands-decider-latest") -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.calls = 0

    # ---- question builders ------------------------------------------------

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no. The answer is P(true). `criteria` describes what true and false mean."""
        q: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            q["criteria"] = dict(criteria)
        return q

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N. `criteria` maps option name -> what that option means."""
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking ------------------------------------------------------------

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """POST /v1/systemone. Returns the full response: answers, usage and latency_ms."""
        body = json.dumps({"state": state, "model": self.model, "questions": dict(questions)}).encode()
        req = urllib.request.Request(f"{self.url}/v1/systemone", data=body,
                                     headers={"content-type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ServerUnavailable(f"No Decider at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}") from exc
        payload["roundtrip_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.calls += 1
        return payload

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ServerUnavailable(f"No Decider at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}") from exc
        self.model = data.get("model", self.model)
        return data
