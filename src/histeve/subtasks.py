"""Parameterized requests the connected VLA is expected to support."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .state import StateSnapshot, evaluate_all


@dataclass(frozen=True)
class Subtask:
    skill_id: str
    family: str
    description: str
    parameter_schema: Mapping[str, Any]
    preconditions: Sequence[Mapping[str, Any]] = ()
    progress_signals: Sequence[str] = ()
    failure_signatures: Sequence[str] = ()
    default_budget_steps: int = 100
    inventory_parameters: Sequence[str] = ()
    completion_criterion: str = ""

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Subtask:
        return cls(**row)

    def normalize_parameters(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        return dict(parameters)

    def execution_parameters(self, parameters, landmarks):
        return dict(parameters)

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        properties = self.parameter_schema.get("properties", {})
        missing = set(self.parameter_schema.get("required", ())) - parameters.keys()
        unknown = parameters.keys() - properties.keys()
        if missing or unknown:
            raise ValueError(f"{self.skill_id}: missing={sorted(missing)}, unknown={sorted(unknown)}")
        types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": Mapping,
            "array": (list, tuple),
        }
        for name, value in parameters.items():
            spec = properties[name]
            kind = spec.get("type")
            if kind in types and (
                not isinstance(value, types[kind])
                or kind in {"integer", "number"}
                and isinstance(value, bool)
            ):
                raise ValueError(f"{self.skill_id}.{name} expects {kind}")
            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(f"{self.skill_id}.{name} is outside enum")
        if "count" in parameters and (type(parameters["count"]) is not int or parameters["count"] < 1):
            raise ValueError("count must be a positive integer")

    def with_visual_inventory(self, state: StateSnapshot) -> Subtask | None:
        """Restrict possession-dependent arguments to items actually observed."""
        schema = copy.deepcopy(dict(self.parameter_schema))
        bound = list(self.inventory_parameters)
        known = {item for item, count in state.inventory.items() if count > 0}
        for name in self.inventory_parameters:
            spec = schema["properties"][name]
            choices = sorted(known.intersection(spec.get("enum", known)))
            if choices:
                spec["enum"] = choices
            elif name in schema.get("required", ()):
                return None
            else:
                del schema["properties"][name]
                bound.remove(name)
        return replace(self, parameter_schema=schema, inventory_parameters=tuple(bound))

    def planning_schema(self, state: StateSnapshot, criterion: str) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "parameter_schema": copy.deepcopy(dict(self.parameter_schema)),
            "applicability": list(self.preconditions),
            "inventory_parameters": list(self.inventory_parameters),
            "progress_signals": list(self.progress_signals),
            "failure_conditions": list(self.failure_signatures),
            "visual_completion_criterion": criterion,
            "budget_steps": self.default_budget_steps,
        }


class Subtasks:
    def __init__(self, subtasks: Iterable[Subtask]):
        self._items = {item.skill_id: item for item in subtasks}
        if not self._items or any(item.default_budget_steps <= 0 for item in self._items.values()):
            raise ValueError("provide subtasks with positive execution budgets")

    @classmethod
    def from_json(cls, path: str | Path) -> Subtasks:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))["subtasks"]
        return cls(Subtask.from_dict(row) for row in rows)

    def ids(self) -> list[str]:
        return sorted(self._items)

    def get(self, skill_id: str) -> Subtask:
        return self._items[skill_id]

    def applicable(self, state: StateSnapshot) -> list[Subtask]:
        result = []
        for name in self.ids():
            item = self.get(name)
            if evaluate_all(item.preconditions, state).met:
                bound = item.with_visual_inventory(state)
                if bound is not None:
                    result.append(bound)
        return result
