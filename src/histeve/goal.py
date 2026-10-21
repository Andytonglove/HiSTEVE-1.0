from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .state import PredicateResult, StateSnapshot, evaluate_all


@dataclass(frozen=True)
class GoalTask:
    task_id: str
    instruction: str
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    goal_predicates: Sequence[Mapping[str, Any]] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, int] = field(default_factory=lambda: {"steps": 3600, "minutes": 20})
    tags: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalTask":
        task = cls(**data)
        task.validate()
        return task

    def validate(self) -> None:
        if not self.task_id or not self.instruction:
            raise ValueError("task_id and instruction cannot be empty")
        if int(self.budget.get("steps", 0)) <= 0:
            raise ValueError(f"{self.task_id} requires a positive step budget")

    def evaluate(self, state: StateSnapshot) -> PredicateResult:
        return evaluate_all(self.goal_predicates, state)

    def agent_payload(self) -> Dict[str, Any]:
        return {"instruction": self.instruction}
