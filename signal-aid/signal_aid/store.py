"""In-memory incident store with a cheap space/time prefilter for deduplication."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass
class Incident:
    id: str
    category: str
    address: str
    brief: str
    lat: float
    lon: float
    priority: bool
    created_at: str
    updated_at: str
    report_ids: list[str] = field(default_factory=list)
    first_hand_reports: int = 0

    @property
    def status(self) -> str:
        # Confirmed when someone on site reported it, or at least two independent reports exist.
        return "confirmed" if self.first_hand_reports > 0 or len(self.report_ids) >= 2 else "unconfirmed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        return d


class IncidentStore:
    def __init__(self) -> None:
        self.incidents: dict[str, Incident] = {}
        self.review_queue: list[dict[str, Any]] = []
        self.volunteer_offers: list[dict[str, Any]] = []

    def publish(self, msg: dict[str, Any], category: str, address: str, brief: str,
                priority: bool, first_hand: bool) -> Incident:
        iid = f"INC-{len(self.incidents) + 1:03d}"
        inc = Incident(iid, category, address, brief, msg["lat"], msg["lon"], priority,
                       msg["ts"], msg["ts"], [msg["id"]], int(first_hand))
        self.incidents[iid] = inc
        return inc

    def merge(self, iid: str, msg: dict[str, Any], brief: str | None, first_hand: bool,
              priority: bool) -> Incident:
        inc = self.incidents[iid]
        inc.report_ids.append(msg["id"])
        inc.first_hand_reports += int(first_hand)
        inc.updated_at = msg["ts"]
        inc.priority = inc.priority or priority
        if brief:
            inc.brief = brief
        return inc

    def nearby(self, msg: dict[str, Any], radius_m: float, window_min: float) -> list[tuple[Incident, float]]:
        out = []
        t = _ts(msg["ts"])
        for inc in self.incidents.values():
            d = haversine_m(msg["lat"], msg["lon"], inc.lat, inc.lon)
            age_min = abs((t - _ts(inc.updated_at)).total_seconds()) / 60
            if d <= radius_m and age_min <= window_min:
                out.append((inc, d))
        return sorted(out, key=lambda x: x[1])[:5]

    def lookup_for(self, radius_m: float, window_min: float):
        """Build the IncidentLookup the TriageGate needs."""
        def lookup(iid: str, msg: dict[str, Any]):
            for inc, d in self.nearby(msg, radius_m, window_min):
                if inc.id == iid:
                    return f"{inc.category} at {inc.address}: {inc.brief}", d
            return None
        return lookup
