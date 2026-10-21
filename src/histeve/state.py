from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Set


def normalize_name(value: Any) -> str:
    return str(value).strip().lower().replace("minecraft:", "")


@dataclass(frozen=True)
class StateSnapshot:
    inventory: Mapping[str, int] = field(default_factory=dict)
    visible: Set[str] = field(default_factory=set)
    recent_events: Sequence[str] = field(default_factory=tuple)
    placed: Mapping[str, int] = field(default_factory=dict)
    flags: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StateSnapshot":
        return cls(
            inventory=_count_map(data.get("inventory", {})),
            visible={normalize_name(item) for item in data.get("visible", data.get("visible_objects", []))},
            recent_events=tuple(normalize_name(item) for item in data.get("recent_events", [])),
            placed=_count_map(data.get("placed", {})),
            flags=dict(data.get("flags", {})),
        )


@dataclass(frozen=True)
class PredicateResult:
    met: bool
    evidence: str


def _count_map(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    result: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            result[normalize_name(key)] = int(value)
        except (TypeError, ValueError):
            result[normalize_name(key)] = 0
    return result


def _items(raw: Any) -> Iterable[str]:
    if isinstance(raw, str):
        return [normalize_name(item) for item in raw.split(",") if item.strip()]
    if isinstance(raw, Sequence):
        return [normalize_name(item) for item in raw]
    return []


def evaluate_predicate(predicate: Mapping[str, Any], state: StateSnapshot) -> PredicateResult:
    if len(predicate) != 1:
        raise ValueError(f"predicate must contain exactly one operator: {predicate}")
    operator, value = next(iter(predicate.items()))

    if operator == "inventory_at_least":
        required = _count_map(value)
        missing = {item: count for item, count in required.items() if state.inventory.get(item, 0) < count}
        return PredicateResult(not missing, f"inventory={dict(state.inventory)}; missing={missing}")

    if operator == "inventory_any_at_least":
        if not isinstance(value, Mapping):
            raise ValueError("inventory_any_at_least expects an object")
        count = int(value.get("count", 1))
        candidates = list(_items(value.get("items", [])))
        matched = {
            item: state.inventory.get(item, 0) for item in candidates if state.inventory.get(item, 0) >= count
        }
        return PredicateResult(
            bool(matched), f"matched_inventory={matched}; required_any={candidates}:{count}"
        )

    if operator == "visible_any":
        candidates = set(_items(value))
        matched = sorted(candidates & {normalize_name(item) for item in state.visible})
        return PredicateResult(bool(matched), f"visible_matches={matched}")

    if operator == "event_any":
        needles = list(_items(value))
        matched = [event for event in state.recent_events if any(needle in event for needle in needles)]
        return PredicateResult(bool(matched), f"event_matches={matched[-4:]}")

    if operator == "placed_at_least":
        required = _count_map(value)
        missing = {item: count for item, count in required.items() if state.placed.get(item, 0) < count}
        return PredicateResult(not missing, f"placed={dict(state.placed)}; missing={missing}")

    if operator == "flag_equals":
        if not isinstance(value, Mapping) or len(value) != 1:
            raise ValueError("flag_equals expects one key/value pair")
        key, expected = next(iter(value.items()))
        actual = state.flags.get(str(key))
        return PredicateResult(actual == expected, f"flag:{key}={actual!r}; expected={expected!r}")

    if operator in {"all_of", "any_of"}:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{operator} expects a list")
        results = [evaluate_predicate(item, state) for item in value]
        met = all(item.met for item in results) if operator == "all_of" else any(item.met for item in results)
        return PredicateResult(met, f"{operator}=[" + "; ".join(item.evidence for item in results) + "]")

    raise ValueError(f"unsupported predicate operator: {operator}")


def evaluate_all(predicates: Sequence[Mapping[str, Any]], state: StateSnapshot) -> PredicateResult:
    results = [evaluate_predicate(predicate, state) for predicate in predicates]
    return PredicateResult(all(item.met for item in results), "; ".join(item.evidence for item in results))
