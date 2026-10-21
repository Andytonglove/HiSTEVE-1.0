from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .interfaces import Frame, VisualAssessment, _no_coordinates
from .state import StateSnapshot

TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "current",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "requested",
    "the",
    "to",
    "using",
    "while",
    "with",
}

TOKEN_ALIASES = {
    "find": ("search", "explore", "locate"),
    "search": ("find", "explore", "locate"),
    "explore": ("find", "search", "locate"),
    "collect": ("acquire", "pickup"),
    "acquire": ("collect", "pickup"),
    "make": ("craft",),
    "craft": ("make",),
    "return": ("revisit", "navigate"),
    "revisit": ("return", "navigate"),
}


def tokenize(text: str) -> List[str]:
    base = [
        token
        for token in TOKEN_RE.findall(text.lower().replace("-", " ").replace("_", " "))
        if token not in STOPWORDS
    ]
    expanded = list(base)
    for token in base:
        expanded.extend(TOKEN_ALIASES.get(token, ()))
    return expanded


FORBIDDEN_LITERAL_PARAMETER_KEYS = {
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "absolute_position",
    "literal_coordinates",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    task_id: str
    seed: str
    skill_id: str
    parameters: Mapping[str, Any]
    status: str
    start_step: int
    end_step: int
    pre_state: Mapping[str, Any] = field(default_factory=dict)
    post_state: Mapping[str, Any] = field(default_factory=dict)
    progress_events: Sequence[str] = field(default_factory=tuple)
    failure_signature: str = ""
    evidence_refs: Sequence[str] = field(default_factory=tuple)
    context_tags: Sequence[str] = field(default_factory=tuple)
    protocol_split: str = "deployment"
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeRecord":
        values = dict(data)
        values.pop("cost_steps", None)
        return cls(**values)

    def validate(self) -> None:
        if self.status not in {"success", "failure", "interrupted"}:
            raise ValueError(f"invalid episode status: {self.status}")
        if self.end_step < self.start_step:
            raise ValueError("episode end_step is before start_step")
        if self.protocol_split not in {"bootstrap", "deployment", "train", "fixture"}:
            raise ValueError(f"invalid protocol_split: {self.protocol_split}")
        if self.status == "failure" and not self.failure_signature:
            raise ValueError("failed episode requires failure_signature")

    @property
    def cost_steps(self) -> int:
        return self.end_step - self.start_step

    def search_text(self) -> str:
        return " ".join(
            [self.task_id, self.skill_id, self.status, self.failure_signature]
            + list(self.progress_events)
            + list(self.context_tags)
            + [str(value) for value in self.parameters.values()]
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cost_steps"] = self.cost_steps
        return data


@dataclass(frozen=True)
class ProcedureStep:
    step_id: str
    skill_id: str
    parameters: Mapping[str, Any]
    dependencies: Sequence[str] = field(default_factory=tuple)
    expected_progress: Sequence[str] = field(default_factory=tuple)

    def validate(self) -> None:
        forbidden = FORBIDDEN_LITERAL_PARAMETER_KEYS & set(self.parameters)
        if forbidden:
            raise ValueError(f"procedure step contains literal coordinates: {sorted(forbidden)}")


@dataclass(frozen=True)
class ProcedureRecord:
    procedure_id: str
    goal_signature: str
    steps: Sequence[ProcedureStep]
    source_episode_ids: Sequence[str]
    context_tags: Sequence[str] = field(default_factory=tuple)
    success_contexts: Sequence[str] = field(default_factory=tuple)
    failure_boundaries: Sequence[str] = field(default_factory=tuple)
    protocol_split: str = "bootstrap"
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcedureRecord":
        values = dict(data)
        values["steps"] = tuple(ProcedureStep(**step) for step in values["steps"])
        return cls(**values)

    def validate(self) -> None:
        ids = [step.step_id for step in self.steps]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("procedure requires unique non-empty steps")
        known = set(ids)
        for step in self.steps:
            step.validate()
            unknown = set(step.dependencies) - known
            if unknown:
                raise ValueError(f"{step.step_id} has unknown dependencies: {sorted(unknown)}")
            if step.step_id in step.dependencies:
                raise ValueError(f"{step.step_id} depends on itself")

    def search_text(self) -> str:
        return " ".join(
            [self.goal_signature]
            + list(self.context_tags)
            + list(self.success_contexts)
            + list(self.failure_boundaries)
            + [step.skill_id for step in self.steps]
            + [str(value) for step in self.steps for value in step.parameters.values()]
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryQuery:
    goal: str
    context_tags: Sequence[str] = field(default_factory=tuple)
    failure_signature: str = ""
    task_id: str = ""
    allowed_protocol_splits: Sequence[str] = ("bootstrap", "train", "fixture")
    excluded_seed: str = ""


@dataclass(frozen=True)
class MemoryMatch:
    record_type: str
    record_id: str
    score: float
    record: Any
    reasons: Sequence[str]


class MemoryIndex:
    def __init__(
        self,
        episodes: Iterable[EpisodeRecord] = (),
        procedures: Iterable[ProcedureRecord] = (),
    ) -> None:
        self.episodes = {item.episode_id: item for item in episodes}
        self.procedures = {item.procedure_id: item for item in procedures}
        for item in self.episodes.values():
            item.validate()
        for item in self.procedures.values():
            item.validate()

    def add_episode(self, episode: EpisodeRecord) -> None:
        episode.validate()
        self.episodes[episode.episode_id] = episode

    def add_procedure(self, procedure: ProcedureRecord) -> None:
        procedure.validate()
        self.procedures[procedure.procedure_id] = procedure

    def retrieve(self, query: MemoryQuery, top_k: int = 5) -> List[MemoryMatch]:
        query_tokens = set(
            tokenize(" ".join([query.goal, query.failure_signature] + list(query.context_tags)))
        )
        allowed_splits = set(query.allowed_protocol_splits)
        matches: List[MemoryMatch] = []

        for episode in self.episodes.values():
            if episode.protocol_split not in allowed_splits:
                continue
            if query.excluded_seed and episode.seed == query.excluded_seed:
                continue
            score, reasons = self._score(
                query_tokens, episode.search_text(), query, episode.task_id, episode.context_tags
            )
            if query.failure_signature and episode.failure_signature == query.failure_signature:
                score += 1
                reasons.append("same_failure_signature")
            if episode.status == "success":
                score += 1
                reasons.append("verified_success")
            matches.append(MemoryMatch("episode", episode.episode_id, score, episode, tuple(reasons)))

        for procedure in self.procedures.values():
            if procedure.protocol_split not in allowed_splits:
                continue
            score, reasons = self._score(
                query_tokens, procedure.search_text(), query, "", procedure.context_tags
            )
            score += 1.0
            reasons.append("verified_procedure")
            matches.append(MemoryMatch("procedure", procedure.procedure_id, score, procedure, tuple(reasons)))

        matches.sort(key=lambda item: (-item.score, item.record_id))
        positive = [item for item in matches if item.score > 0]
        return (positive or matches)[: max(0, top_k)]

    @staticmethod
    def _score(
        query_tokens: set[str],
        search_text: str,
        query: MemoryQuery,
        task_id: str,
        context_tags: Sequence[str],
    ) -> Tuple[float, List[str]]:
        document_tokens = set(tokenize(search_text))
        overlap = query_tokens & document_tokens
        score = len(overlap)
        reasons = [f"token_overlap={sorted(overlap)}"] if overlap else []
        tag_overlap = set(query.context_tags) & set(context_tags)
        if tag_overlap:
            score += len(tag_overlap)
            reasons.append(f"tag_overlap={sorted(tag_overlap)}")
        if query.task_id and query.task_id == task_id:
            score += 1
            reasons.append("same_task")
        return score, reasons


def extract_linear_procedure(
    procedure_id: str,
    goal_signature: str,
    episodes: Sequence[EpisodeRecord],
    context_tags: Sequence[str] = (),
) -> ProcedureRecord:
    successful = sorted(
        (episode for episode in episodes if episode.status == "success"),
        key=lambda episode: (episode.start_step, episode.end_step, episode.episode_id),
    )
    if not successful:
        raise ValueError("cannot extract a procedure without successful episodes")
    source_splits = {episode.protocol_split for episode in successful}
    if len(source_splits) != 1:
        raise ValueError(f"cannot mix protocol splits in one procedure: {sorted(source_splits)}")
    source_split = next(iter(source_splits))
    steps: List[ProcedureStep] = []
    previous = ""
    for index, episode in enumerate(successful):
        step_id = f"step_{index:03d}_{episode.skill_id}"
        steps.append(
            ProcedureStep(
                step_id=step_id,
                skill_id=episode.skill_id,
                parameters=dict(episode.parameters),
                dependencies=(previous,) if previous else (),
                expected_progress=tuple(episode.progress_events),
            )
        )
        previous = step_id
    procedure = ProcedureRecord(
        procedure_id=procedure_id,
        goal_signature=goal_signature,
        steps=tuple(steps),
        source_episode_ids=tuple(episode.episode_id for episode in successful),
        context_tags=tuple(context_tags),
        success_contexts=tuple(sorted({tag for episode in successful for tag in episode.context_tags})),
        failure_boundaries=tuple(
            sorted({episode.failure_signature for episode in episodes if episode.failure_signature})
        ),
        protocol_split=source_split,
    )
    procedure.validate()
    return procedure


class Memory:
    """Episode state, landmarks and experience retrieval."""

    def __init__(self, run_id: str, seed: str, history: MemoryIndex | None = None):
        if not run_id:
            raise ValueError("run_id is required")
        self.run_id, self.seed = run_id, seed
        self.history = history or MemoryIndex()
        self.current: list[EpisodeRecord] = []
        self.world: dict[str, str] = {}
        self.landmark_frames: dict[str, tuple[Frame, ...]] = {}
        self.procedures: list[ProcedureRecord] = []

    def observe(self, assessment: VisualAssessment) -> VisualAssessment:
        for name, description in assessment.landmarks.items():
            if description != self.world.get(name):
                self.landmark_frames[name] = assessment.observation_frames[-1:]
            self.world[name] = description
        return replace(assessment, landmarks=dict(self.world), landmark_frames=dict(self.landmark_frames))

    @staticmethod
    def state_tags(state: StateSnapshot) -> tuple[str, ...]:
        """Build retrieval tags from the observed state."""
        tags = set(state.visible) | set(state.recent_events)
        tags.update(key for key, count in state.inventory.items() if count > 0)
        tags.update(key for key, count in state.placed.items() if count > 0)
        tags.update(
            f"flag:{key}={str(value).lower()}" for key, value in state.flags.items() if type(value) is bool
        )
        return tuple(sorted(tags))

    def commit(self, record: EpisodeRecord, goal: str) -> None:
        if not record.evidence_refs:
            raise ValueError("memory requires verified visual evidence")
        self.current.append(record)
        if record.status == "success":
            # Build a procedure from the latest consecutive successes.
            chain = []
            for row in reversed(self.current):
                if row.status != "success":
                    break
                chain.append(row)
            _no_coordinates([row.parameters for row in chain])
            procedure = extract_linear_procedure(
                f"{self.run_id}:procedure:{len(self.procedures)}",
                goal,
                list(reversed(chain)),
                context_tags=chain[-1].context_tags,
            )
            if len(chain) > 1:
                self.procedures[-1] = procedure
            else:
                self.procedures.append(procedure)

    def retrieve(
        self, goal: str, *, state: StateSnapshot | None = None, failure_signature: str = "", task_id: str = ""
    ) -> list[dict[str, Any]]:
        # Retrieve historical experience from training and bootstrap runs.
        episodes = [
            row
            for row in self.history.episodes.values()
            if row.evidence_refs and row.seed != self.seed and row.protocol_split in {"train", "bootstrap"}
        ]
        for row in episodes:
            _no_coordinates(row.parameters)
        sources = {row.episode_id: row for row in episodes}
        procedures: list[ProcedureRecord] = []
        for procedure in self.history.procedures.values():
            if not procedure.source_episode_ids or procedure.protocol_split not in {"train", "bootstrap"}:
                continue
            if not all(
                key in sources and sources[key].status == "success" for key in procedure.source_episode_ids
            ):
                continue
            _no_coordinates([step.parameters for step in procedure.steps])
            procedures.append(procedure)
        index = MemoryIndex(episodes, procedures)
        query = MemoryQuery(
            goal,
            context_tags=self.state_tags(state) if state is not None else (),
            failure_signature=failure_signature,
            task_id=task_id,
            excluded_seed=self.seed,
        )
        matches = index.retrieve(query, top_k=len(episodes) + len(procedures))
        matches = [
            match
            for match in matches
            if any(
                reason.startswith(("token_overlap=", "tag_overlap="))
                or reason in {"same_task", "same_failure_signature"}
                for reason in match.reasons
            )
        ][:4]
        records = [match.record.to_dict() for match in matches]
        # Include experience from the current episode.
        local_query = replace(
            query, excluded_seed="", allowed_protocol_splits=("deployment", "train", "bootstrap")
        )
        current_matches = MemoryIndex(self.current).retrieve(local_query, top_k=len(self.current))
        current_matches.sort(key=lambda match: (-match.score, -match.record.end_step, match.record_id))
        for row in self.current:
            _no_coordinates(row.parameters)
        return (
            records
            + [match.record.to_dict() for match in current_matches[:4]]
            + [row.to_dict() for row in self.procedures[-2:]]
        )
