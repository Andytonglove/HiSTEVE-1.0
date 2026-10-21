from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
from histeve.goal import GoalTask
from histeve.state import StateSnapshot, evaluate_all


@dataclass(frozen=True)
class DiagnosticNode:
    node_id: str
    description: str
    predicates: Sequence[Mapping[str, Any]]
    dependencies: Sequence[str] = field(default_factory=tuple)
    dependency_mode: str = "all"
    requires_prior_dependencies: bool = False


@dataclass(frozen=True)
class DiagnosticProgress:
    reached: Set[str]
    newly_reached: Set[str]
    frontier: Set[str]
    completion_ratio: float
    evidence: Mapping[str, str]


@dataclass(frozen=True)
class DiagnosticGraph:
    task_id: str
    nodes: Sequence[DiagnosticNode]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagnosticGraph":
        graph = cls(
            task_id=str(data["task_id"]),
            nodes=tuple(
                (
                    DiagnosticNode(
                        node_id=str(row["node_id"]),
                        description=str(row.get("description", "")),
                        predicates=tuple(row.get("predicates", [])),
                        dependencies=tuple((str(item) for item in row.get("dependencies", []))),
                        dependency_mode=str(row.get("dependency_mode", "all")),
                        requires_prior_dependencies=bool(row.get("requires_prior_dependencies", False)),
                    )
                    for row in data.get("nodes", [])
                )
            ),
        )
        graph.validate()
        return graph

    def validate(self) -> None:
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate diagnostic node in {self.task_id}")
        known = set(ids)
        for node in self.nodes:
            unknown = set(node.dependencies) - known
            if unknown:
                raise ValueError(f"{node.node_id} has unknown dependencies: {sorted(unknown)}")
            if node.node_id in node.dependencies:
                raise ValueError(f"{node.node_id} depends on itself")
            if node.dependency_mode not in {"all", "any"}:
                raise ValueError(f"invalid dependency_mode: {node.dependency_mode}")
            if node.requires_prior_dependencies and (not node.dependencies):
                raise ValueError(f"{node.node_id} requires prior dependencies but has none")
        visiting: Set[str] = set()
        visited: Set[str] = set()
        dependencies = {node.node_id: set(node.dependencies) for node in self.nodes}

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError(f"diagnostic graph contains a cycle at {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in dependencies[node_id]:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in ids:
            visit(node_id)

    def evaluate(self, state: StateSnapshot, reached: Optional[Set[str]] = None) -> DiagnosticProgress:
        reached_now = set(reached or set())
        reached_before_observation = set(reached_now)
        newly_reached: Set[str] = set()
        evidence: Dict[str, str] = {}
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                if node.node_id in reached_now:
                    continue
                dependency_evidence = (
                    reached_before_observation if node.requires_prior_dependencies else reached_now
                )
                dependencies_met = (
                    not node.dependencies
                    or (node.dependency_mode == "all" and set(node.dependencies) <= dependency_evidence)
                    or (node.dependency_mode == "any" and bool(set(node.dependencies) & dependency_evidence))
                )
                if not dependencies_met:
                    continue
                result = evaluate_all(node.predicates, state)
                if result.met:
                    reached_now.add(node.node_id)
                    newly_reached.add(node.node_id)
                    evidence[node.node_id] = result.evidence
                    changed = True
        frontier = {
            node.node_id
            for node in self.nodes
            if node.node_id not in reached_now
            and (
                not node.dependencies
                or (
                    node.dependency_mode == "all"
                    and set(node.dependencies)
                    <= (reached_before_observation if node.requires_prior_dependencies else reached_now)
                )
                or (
                    node.dependency_mode == "any"
                    and bool(
                        set(node.dependencies)
                        & (reached_before_observation if node.requires_prior_dependencies else reached_now)
                    )
                )
            )
        }
        ratio = len(reached_now) / len(self.nodes) if self.nodes else 1.0
        return DiagnosticProgress(reached_now, newly_reached, frontier, ratio, evidence)


class LongBenchCatalog:
    def __init__(self, tasks: Sequence[GoalTask], diagnostics: Sequence[DiagnosticGraph]):
        self.tasks = {task.task_id: task for task in tasks}
        self.diagnostics = {graph.task_id: graph for graph in diagnostics}
        if len(self.tasks) != len(tasks):
            raise ValueError("duplicate task_id in public catalog")
        if len(self.diagnostics) != len(diagnostics):
            raise ValueError("duplicate task_id in diagnostic catalog")
        missing = set(self.tasks) - set(self.diagnostics)
        if missing:
            raise ValueError(f"tasks missing diagnostic graphs: {sorted(missing)}")

    @classmethod
    def from_json(cls, public_path: str | Path, diagnostic_path: str | Path) -> "LongBenchCatalog":
        public_data = json.loads(Path(public_path).read_text(encoding="utf-8"))
        diagnostic_data = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))
        return cls(
            tasks=[GoalTask.from_dict(row) for row in public_data["tasks"]],
            diagnostics=[DiagnosticGraph.from_dict(row) for row in diagnostic_data["diagnostics"]],
        )

    def task(self, task_id: str) -> GoalTask:
        return self.tasks[task_id]

    def diagnostic(self, task_id: str) -> DiagnosticGraph:
        return self.diagnostics[task_id]

    def agent_catalog(self) -> List[Dict[str, Any]]:
        return [self.tasks[task_id].agent_payload() for task_id in sorted(self.tasks)]
