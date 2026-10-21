"""Data and interfaces exchanged by the planner, harness, models and environment."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from .state import StateSnapshot


@dataclass(frozen=True)
class Frame:
    step: int
    image_ref: str
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.step < 0 or not self.image_ref:
            raise ValueError("a frame requires a nonnegative step and an image reference")


BUTTONS = frozenset(
    ("forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "drop", "inventory")
    + tuple(f"hotbar.{n}" for n in range(1, 10))
)


@dataclass(frozen=True)
class NativeAction:
    buttons: tuple[str, ...] = ()
    camera: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not set(self.buttons) <= BUTTONS or len(set(self.buttons)) != len(self.buttons):
            raise ValueError("only native keyboard/mouse buttons are allowed")
        if len(self.camera) != 2 or not all(math.isfinite(x) and abs(x) <= 180 for x in self.camera):
            raise ValueError("camera must contain two finite native displacements in [-180, 180]")


@dataclass(frozen=True)
class VisualGoalAssessment:
    status: str
    reason: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class VisualAssessment:
    state: StateSnapshot
    evidence_refs: tuple[str, ...]
    status: str = "interrupted"
    reason: str = ""
    recoverable: bool = False
    # Landmark appearance and relative surroundings.
    landmarks: Mapping[str, str] = field(default_factory=dict)
    # Frames attached by the runtime.
    observation_frames: tuple[Frame, ...] = ()
    goal: VisualGoalAssessment | None = None
    inventory_scope: str = "unknown"
    inventory_gains: Mapping[str, int] = field(default_factory=dict)
    landmark_frames: Mapping[str, tuple[Frame, ...]] = field(default_factory=dict)


INVENTORY_SCOPES = frozenset(("storage_total", "hotbar_lower_bound", "unknown"))


def inventory_deltas(before: VisualAssessment, after: VisualAssessment) -> dict[str, int]:
    """Prefer comparable storage counts; otherwise use witnessed item transfers."""
    gains = dict(after.inventory_gains)
    if before.inventory_scope == after.inventory_scope == "storage_total":
        for item in before.state.inventory.keys() | after.state.inventory.keys():
            delta = after.state.inventory.get(item, 0) - before.state.inventory.get(item, 0)
            gains.pop(item, None)
            if delta > 0:
                gains[item] = delta
    return gains


def assessment_state(value: VisualAssessment) -> dict[str, Any]:
    return {**asdict(value.state), "inventory_scope": value.inventory_scope}


@dataclass(frozen=True)
class Command:
    skill_id: str
    parameters: Mapping[str, Any]


class InvalidPlannerResponse(ValueError):
    """Invalid planner output."""


class InvalidVisualResponse(ValueError):
    """Invalid visual assessment."""


class VisualEnvironment(Protocol):
    def step(self, action: NativeAction) -> Frame: ...


class ActionModel(Protocol):
    """The external VLA model used by the default Minecraft integration."""

    def reset(self) -> None: ...
    def set_route(self, *, is_gui: bool, needs_table: bool = False) -> None: ...
    def act(self, frame: Frame, instruction: str) -> NativeAction: ...
    def policy_context(self, frame: Frame, instruction: str) -> Any: ...


class DetectorModel(Protocol):
    """The external state detector; policy_context is an opaque model value."""

    def predict(
        self, history: Sequence[Frame], current: Frame, instruction: str, policy_context: Any
    ) -> float: ...


class Policy(Protocol):
    def begin(self, instruction: str) -> None: ...
    def act(self, frame: Frame, instruction: str) -> NativeAction: ...


class VisualVerifier(Protocol):
    def assess(
        self, frames: Sequence[Frame], command: Command | None, before: VisualAssessment | None
    ) -> VisualAssessment: ...


class Planner(Protocol):
    def choose(self, context: Mapping[str, Any]) -> Command: ...


@dataclass(frozen=True)
class DetectorPrediction:
    query_id: int
    probability: float


class CompletionDetector(Protocol):
    def reset(self, instruction: str) -> None: ...
    def predict(self, frame: Frame, instruction: str) -> DetectorPrediction | None: ...


@dataclass(frozen=True)
class RuntimeConfig:
    handoff: str = "detector"
    threshold: float = 0.7
    history_frames: int = 4
    max_local_retries: int = 1

    def __post_init__(self) -> None:
        if self.handoff not in {"detector", "fixed"}:
            raise ValueError("handoff must be detector or fixed")
        if not 0 <= self.threshold <= 1 or self.history_frames < 1 or self.max_local_retries < 0:
            raise ValueError("invalid runtime configuration")


def public_frame(frame: Frame) -> dict[str, Any]:
    return {"step": frame.step, "image_ref": frame.image_ref}


def validate_assessment(
    value: VisualAssessment,
    frames: Sequence[Frame],
    before_frames: Sequence[Frame] = (),
    *,
    require_goal=False,
) -> VisualAssessment:
    """Check model evidence at the boundary before it becomes planning state."""
    current = {frame.image_ref for frame in frames}
    observed = current | {frame.image_ref for frame in before_frames}

    def check_refs(refs):
        if not isinstance(refs, (list, tuple)) or not all(isinstance(ref, str) for ref in refs):
            raise InvalidVisualResponse("evidence references must be strings")
        if not set(refs) & current or not set(refs) <= observed:
            raise InvalidVisualResponse("assessment must cite current evidence and only supplied frames")

    if not isinstance(value.status, str) or value.status not in {"success", "failure", "interrupted"}:
        raise InvalidVisualResponse("invalid visual outcome")
    if not isinstance(value.inventory_scope, str) or value.inventory_scope not in INVENTORY_SCOPES:
        raise InvalidVisualResponse("invalid inventory scope")
    if value.inventory_scope == "unknown" and value.state.inventory:
        raise InvalidVisualResponse("unknown inventory must have no reported counts")
    check_refs(value.evidence_refs)
    if frames[-1].image_ref not in value.evidence_refs:
        raise InvalidVisualResponse("assessment must cite the latest current frame")
    for counts in (value.state.inventory, value.state.placed, value.inventory_gains):
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise InvalidVisualResponse("visual counts must be nonnegative integers")
    if value.inventory_gains and not set(value.evidence_refs).intersection(
        frame.image_ref for frame in before_frames
    ):
        raise InvalidVisualResponse("item transfers require before and current evidence")
    if value.goal is None:
        if require_goal:
            raise InvalidVisualResponse("public goal assessment is required")
    else:
        if (
            not isinstance(value.goal.status, str)
            or value.goal.status not in {"complete", "incomplete", "uncertain"}
            or not isinstance(value.goal.reason, str)
            or not value.goal.reason.strip()
        ):
            raise InvalidVisualResponse("invalid public goal assessment")
        check_refs(value.goal.evidence_refs)
    if not isinstance(value.landmarks, Mapping) or not all(
        isinstance(v, str) for v in value.landmarks.values()
    ):
        raise InvalidVisualResponse("landmarks must be visual descriptions")
    try:
        _no_coordinates(value.landmarks)
    except ValueError as exc:
        raise InvalidVisualResponse(str(exc)) from exc
    return replace(value, observation_frames=tuple(frames))


def _no_coordinates(value: Any) -> None:
    forbidden = {
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
        "position",
        "pose",
        "absolute_position",
        "literal_coordinates",
    }
    if isinstance(value, Mapping):
        if forbidden & set(value):
            raise ValueError("procedural memory cannot contain literal pose/coordinates")
        for item in value.values():
            _no_coordinates(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _no_coordinates(item)
    elif isinstance(value, str):
        # Check free-text descriptions for coordinates and fixed orientations.
        literal_axis = re.search(r"\b(?:x|y|z|yaw|pitch)\s*[:=]\s*-?\d+(?:\.\d+)?\b", value, re.I)
        literal_pose = re.search(r"\b(?:position|pose|coordinates?)\b\s*[:=]?\s*[\[(]?\s*-?\d", value, re.I)
        numeric_tuple = re.search(
            r"[\[(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*[\])]",
            value,
        )
        fixed_orientation = re.search(
            r"\b(?:face|facing|look(?:ing)?|camera)\s+(?:north|south|east|west)\b",
            value,
            re.I,
        )
        numeric_orientation = re.search(
            r"\b(?:face|facing|look(?:ing)?|camera)\s+(?:(?:up|down|left|right)\s+)?"
            r"-?\d+(?:\.\d+)?\s*(?:degrees?|deg)\b",
            value,
            re.I,
        )
        if literal_axis or literal_pose or numeric_tuple or fixed_orientation or numeric_orientation:
            raise ValueError("procedural memory cannot contain literal pose/coordinates")
