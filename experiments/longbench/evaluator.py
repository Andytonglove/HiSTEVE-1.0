from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
from histeve.state import StateSnapshot, normalize_name
from histeve.goal import GoalTask
from .catalog import DiagnosticGraph, DiagnosticProgress

JsonDict = Dict[str, Any]


@dataclass
class Observation:
    step: int
    current_frame_path: Optional[str] = None
    structured: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class LandmarkRecord:
    landmark_id: str
    description: str
    pose: Mapping[str, float]
    evidence_refs: Sequence[str]
    confidence: float
    observed_step: int
    ttl_steps: int = 6000

    def expired(self, current_step: int) -> bool:
        return current_step - self.observed_step > self.ttl_steps


class LandmarkStore:
    def __init__(self) -> None:
        self._landmarks: Dict[str, LandmarkRecord] = {}

    def remember(self, landmark: LandmarkRecord) -> None:
        if not landmark.evidence_refs:
            raise ValueError("landmark requires visual or event evidence")
        if not 0.0 <= landmark.confidence <= 1.0:
            raise ValueError("landmark confidence must be in [0, 1]")
        self._landmarks[landmark.landmark_id] = landmark

    def retrieve(self, landmark_id: str, current_step: int) -> Optional[LandmarkRecord]:
        landmark = self._landmarks.get(landmark_id)
        if landmark is None or landmark.expired(current_step):
            return None
        return landmark

    def active(self, current_step: int) -> List[LandmarkRecord]:
        return sorted(
            (item for item in self._landmarks.values() if not item.expired(current_step)),
            key=lambda item: (-item.confidence, -item.observed_step, item.landmark_id),
        )


@dataclass(frozen=True)
class LandmarkPolicy:
    aliases: Sequence[str]
    leave_distance: float = 12.0
    return_distance: float = 3.0
    require_hidden_to_leave: bool = False


LANDMARK_POLICIES: Mapping[str, LandmarkPolicy] = {
    "cave_entrance": LandmarkPolicy(("cave", "cave_entrance", "stone", "deepslate")),
    "tree_site": LandmarkPolicy(("tree", "trunk", "log", "oak_log", "birch_log", "spruce_log")),
    "stone_site": LandmarkPolicy(("stone", "cobblestone", "deepslate", "rock")),
    "camp": LandmarkPolicy(("crafting_table", "workbench"), require_hidden_to_leave=True),
}


class LongBenchEvidenceMonitor:
    """Derive benchmark flags from observations and active subtasks.

    Landmark returns require departure, spatial arrival and matching visual evidence.
    """

    def __init__(
        self,
        *,
        task_id: str = "",
        requested_return_landmark: str = "",
        world_memory: LandmarkStore | None = None,
    ) -> None:
        self.task_id = task_id
        self.requested_return_landmark = normalize_name(requested_return_landmark)
        self.world_memory = world_memory or LandmarkStore()
        self._flags: Dict[str, Any] = {}
        self._departed: Dict[str, LandmarkRecord] = {}
        self._entered_cave_anchor: LandmarkRecord | None = None
        self._active_skill = ""
        self._active_parameters: Dict[str, Any] = {}
        self._recorded_in_active_skill = False
        self._last_inventory: Dict[str, int] = {}
        self._pending_noncoal_fuel = ""
        self._smelt_input_committed = False
        self._coal_seen = False
        self._pending_missed_log_step: int | None = None

    @property
    def flags(self) -> Mapping[str, Any]:
        return dict(self._flags)

    def begin_skill(self, skill_id: str, parameters: Mapping[str, Any]) -> None:
        self._active_skill = normalize_name(skill_id)
        self._active_parameters = dict(parameters)
        self._recorded_in_active_skill = False

    def end_skill(self) -> None:
        self._active_skill = ""
        self._active_parameters = {}

    def update(self, observation: Observation) -> Mapping[str, Any]:
        structured = observation.structured or {}
        visible = {
            normalize_name(item).replace(" ", "_") for item in structured.get("visible_objects", []) if item
        }
        pose = self._pose(structured.get("pose"))
        events = tuple(
            (normalize_name(item).replace(" ", "_") for item in structured.get("recent_events", []) if item)
        )
        self._consume_explicit_events(events)
        self._update_controlled_task_evidence(structured, events, observation.step)
        if self._active_skill == "record_landmark" and (not self._recorded_in_active_skill):
            landmark_id = normalize_name(self._active_parameters.get("landmark_id", "")).replace(" ", "_")
            self._record_landmark(landmark_id, observation, pose, visible, events)
        if pose is not None:
            for landmark in self.world_memory.active(observation.step):
                self._update_departure_and_return(landmark, pose, visible)
            self._update_cave_transition(pose, visible, observation.step)
        return self.flags

    def _update_controlled_task_evidence(
        self, structured: Mapping[str, Any], events: Sequence[str], step: int
    ) -> None:
        """Track alternative fuel use and recovery of missed log drops."""
        raw_inventory = structured.get("inventory")
        inventory: Dict[str, int] = {}
        if isinstance(raw_inventory, Mapping):
            for raw_item, raw_count in raw_inventory.items():
                try:
                    inventory[normalize_name(raw_item)] = int(raw_count or 0)
                except (TypeError, ValueError):
                    continue
        raw_delta = structured.get("inventory_delta")
        delta: Dict[str, int] = {}
        if isinstance(raw_delta, Mapping):
            for raw_item, raw_count in raw_delta.items():
                try:
                    delta[normalize_name(raw_item)] = int(raw_count or 0)
                except (TypeError, ValueError):
                    continue
        if not delta and self._last_inventory:
            for item in set(inventory) | set(self._last_inventory):
                change = inventory.get(item, 0) - self._last_inventory.get(item, 0)
                if change:
                    delta[item] = change
        if self.task_id == "fuel_alternative_v1":
            if inventory.get("coal", 0) > 0 or inventory.get("coal_block", 0) > 0:
                self._coal_seen = True
            if any((delta.get(item, 0) < 0 for item in ("raw_iron", "iron_ore"))):
                self._smelt_input_committed = True
            noncoal_fuels = (
                "oak_planks",
                "birch_planks",
                "spruce_planks",
                "jungle_planks",
                "acacia_planks",
                "dark_oak_planks",
                "mangrove_planks",
                "planks",
                "charcoal",
                "stick",
                "lava_bucket",
            )
            for fuel in noncoal_fuels:
                if delta.get(fuel, 0) < 0:
                    self._pending_noncoal_fuel = fuel
                    break
            if (
                delta.get("iron_ingot", 0) > 0
                and self._smelt_input_committed
                and self._pending_noncoal_fuel
                and (not self._coal_seen)
            ):
                self._flags["non_coal_fuel_used"] = True
                self._flags["last_smelt_fuel"] = self._pending_noncoal_fuel
                self._flags["non_coal_fuel_evidence"] = "input+fuel decrease followed by iron_ingot increase"
        if self.task_id == "lost_drop_recovery_v1":
            log_aliases = {
                "oak_log",
                "birch_log",
                "spruce_log",
                "jungle_log",
                "acacia_log",
                "dark_oak_log",
                "mangrove_log",
                "log",
            }
            mined_log = any(
                (
                    event.split("=", 1)[0].partition(":")[0] == "mine_block"
                    and event.split("=", 1)[0].partition(":")[2] in log_aliases
                    for event in events
                )
            )
            picked_log = any(
                (
                    event.split("=", 1)[0].partition(":")[0] == "pickup"
                    and event.split("=", 1)[0].partition(":")[2] in log_aliases
                    for event in events
                )
            ) or any((delta.get(item, 0) > 0 for item in log_aliases))
            if self._pending_missed_log_step is not None and step > self._pending_missed_log_step:
                self._flags["missed_drop_observed"] = True
                self._flags["missed_drop_evidence"] = (
                    f"log mined at step {self._pending_missed_log_step} without same-observation pickup"
                )
                if picked_log:
                    self._flags["missed_drop_recovered"] = True
            if mined_log and (not picked_log):
                self._pending_missed_log_step = step
        self._last_inventory = inventory

    def _record_landmark(
        self,
        landmark_id: str,
        observation: Observation,
        pose: Optional[Mapping[str, float]],
        visible: Set[str],
        events: Sequence[str],
    ) -> None:
        policy = LANDMARK_POLICIES.get(landmark_id)
        if not landmark_id or policy is None or pose is None or (not self._matches(policy, visible)):
            return
        evidence = []
        if observation.current_frame_path:
            evidence.append(observation.current_frame_path)
        evidence.extend((event for event in events if self._event_supports_landmark(event, policy)))
        if not evidence:
            return
        self.world_memory.remember(
            LandmarkRecord(
                landmark_id=landmark_id,
                description=str(self._active_parameters.get("description", landmark_id)),
                pose=dict(pose),
                evidence_refs=tuple(dict.fromkeys(evidence)),
                confidence=0.9,
                observed_step=observation.step,
            )
        )
        self._recorded_in_active_skill = True
        self._flags[f"landmark_recorded:{landmark_id}"] = True
        alias_flags = {
            "cave_entrance": "cave_entrance_recorded",
            "tree_site": "tree_site_recorded",
            "stone_site": "stone_site_recorded",
        }
        if landmark_id in alias_flags:
            self._flags[alias_flags[landmark_id]] = True

    def _update_departure_and_return(
        self, landmark: LandmarkRecord, pose: Mapping[str, float], visible: Set[str]
    ) -> None:
        policy = LANDMARK_POLICIES.get(landmark.landmark_id)
        if policy is None:
            return
        distance = self._horizontal_distance(landmark.pose, pose)
        currently_visible = self._matches(policy, visible)
        if distance >= policy.leave_distance and (
            not policy.require_hidden_to_leave or not currently_visible
        ):
            self._departed[landmark.landmark_id] = landmark
            self._flags[f"landmark_departed:{landmark.landmark_id}"] = True
            if landmark.landmark_id == "camp":
                self._flags["left_camp_visual_range"] = True
        if self._departed.get(landmark.landmark_id) is not landmark:
            return
        if distance > policy.return_distance or not currently_visible:
            return
        self._flags[f"landmark_returned:{landmark.landmark_id}"] = True
        if landmark.landmark_id == "camp":
            self._flags["returned_to_same_camp"] = True
        if landmark.landmark_id == self.requested_return_landmark:
            self._flags["returned_to_requested_resource"] = True

    def _update_cave_transition(self, pose: Mapping[str, float], visible: Set[str], step: int) -> None:
        entrance = self.world_memory.retrieve("cave_entrance", step)
        if entrance is None:
            return
        try:
            vertical_drop = float(entrance.pose["y"]) - float(pose["y"])
        except (KeyError, TypeError, ValueError):
            return
        horizontal = self._horizontal_distance(entrance.pose, pose)
        cave_context = bool(visible & {"cave", "stone", "deepslate", "coal_ore", "iron_ore"})
        if vertical_drop >= 4.0 and horizontal <= 32.0 and cave_context:
            self._flags["cave_entered"] = True
            self._entered_cave_anchor = entrance
        if self._entered_cave_anchor is not entrance:
            return
        near_entry_height = abs(float(entrance.pose["y"]) - float(pose["y"])) <= 2.5
        if (
            horizontal <= LANDMARK_POLICIES["cave_entrance"].return_distance
            and near_entry_height
            and cave_context
        ):
            self._flags["returned_to_cave_entrance"] = True
            self._flags["landmark_returned:cave_entrance"] = True

    def _consume_explicit_events(self, events: Sequence[str]) -> None:
        for event in events:
            kind, _, item = event.partition(":")
            item = item.split("=", 1)[0]
            if kind in {"tool_break", "item_break", "break_item"}:
                self._flags["tool_break_observed"] = True
            elif kind == "smelt_fuel" and item not in {"", "coal", "coal_block"}:
                self._flags["non_coal_fuel_used"] = True
            elif kind in {"missed_drop", "drop_missed"}:
                self._flags["missed_drop_observed"] = True

    @staticmethod
    def _event_supports_landmark(event: str, policy: LandmarkPolicy) -> bool:
        return any((alias in event for alias in policy.aliases))

    @staticmethod
    def _matches(policy: LandmarkPolicy, visible: Set[str]) -> bool:
        return bool(set(policy.aliases) & visible)

    @staticmethod
    def _pose(raw: Any) -> Optional[Mapping[str, float]]:
        if not isinstance(raw, Mapping):
            return None
        pose: Dict[str, float] = {}
        for key in ("x", "y", "z"):
            try:
                pose[key] = float(raw[key])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(pose[key]):
                return None
        return pose

    @staticmethod
    def _horizontal_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
        try:
            dx = float(first["x"]) - float(second["x"])
            dz = float(first["z"]) - float(second["z"])
        except (KeyError, TypeError, ValueError):
            return math.inf
        return math.hypot(dx, dz)


COUNTED_EVENT_RE = re.compile("^(?P<kind>[a-z_]+):(?P<item>[^=]+?)(?:=(?P<count>-?\\d+))?$")


@dataclass(frozen=True)
class EvaluationSnapshot:
    task_id: str
    step: int
    goal_met: bool
    goal_evidence: str
    diagnostic: DiagnosticProgress


class MinecraftStateAdapter:
    """Accumulate benchmark state from observations.

    MineStudio event counters are cumulative; process each counter increase once.
    """

    def __init__(self, evidence_monitor: Any = None) -> None:
        self._event_counts: Dict[str, int] = {}
        self._placed: Dict[str, int] = {}
        self._flags: Dict[str, Any] = {}
        self._last_inventory: Dict[str, int] = {}
        self._new_events: list[str] = []
        self._explicit_placed_items: Set[str] = set()
        self.evidence_monitor = evidence_monitor

    def update(self, observation: Observation) -> StateSnapshot:
        structured = dict(observation.structured or {})
        inventory = self._count_map(structured.get("inventory", {}))
        visible = {normalize_name(item) for item in structured.get("visible_objects", []) if item}
        self._new_events = []
        explicit_placed = structured.get("placed_blocks") or structured.get("placed") or {}
        explicit_placed_counts = self._count_map(explicit_placed)
        self._explicit_placed_items = set(explicit_placed_counts)
        for item, count in explicit_placed_counts.items():
            self._placed[item] = max(self._placed.get(item, 0), count)
        for raw_event in structured.get("recent_events") or []:
            self._consume_event(str(raw_event))
        if self.evidence_monitor is not None:
            monitor_flags = self.evidence_monitor.update(observation)
            if isinstance(monitor_flags, Mapping):
                self._flags.update({str(key): value for key, value in monitor_flags.items()})
        for item, count in inventory.items():
            if count <= self._last_inventory.get(item, 0):
                continue
            if item == "charcoal":
                self._flags["charcoal_produced"] = True
        self._last_inventory = inventory
        return StateSnapshot(
            inventory=inventory,
            visible=visible,
            recent_events=tuple(self._new_events),
            placed=dict(self._placed),
            flags=dict(self._flags),
        )

    def _consume_event(self, raw_event: str) -> None:
        event = normalize_name(raw_event)
        match = COUNTED_EVENT_RE.match(event)
        if not match:
            if event not in self._event_counts:
                self._event_counts[event] = 1
                self._new_events.append(event)
                self._apply_event(event, 1)
            return
        kind = match.group("kind")
        item = normalize_name(match.group("item"))
        key = f"{kind}:{item}"
        count = int(match.group("count") or 1)
        previous = self._event_counts.get(key, 0)
        if count <= previous:
            return
        delta = count - previous
        self._event_counts[key] = count
        normalized_event = f"{key}={delta}"
        self._new_events.append(normalized_event)
        self._apply_event(key, delta)

    def _apply_event(self, event: str, delta: int) -> None:
        kind, _, item = event.partition(":")
        if kind in {"place_block", "placed_block"} and item and (item not in self._explicit_placed_items):
            self._placed[item] = self._placed.get(item, 0) + max(0, delta)
        if kind in {"tool_break", "item_break", "break_item"}:
            self._flags["tool_break_observed"] = True
            if item:
                self._flags["broken_tool"] = item
        if kind in {"smelt_item", "smelt_output"} and item == "charcoal":
            self._flags["charcoal_produced"] = True
        if kind == "smelt_fuel" and item and (item not in {"coal", "coal_block"}):
            self._flags["non_coal_fuel_used"] = True
            self._flags["last_smelt_fuel"] = item
        flag_events = {
            "cave_entrance_recorded": "cave_entrance_recorded",
            "cave_entered": "cave_entered",
            "returned_to_cave_entrance": "returned_to_cave_entrance",
            "tree_site_recorded": "tree_site_recorded",
            "stone_site_recorded": "stone_site_recorded",
            "returned_to_requested_resource": "returned_to_requested_resource",
            "left_camp_visual_range": "left_camp_visual_range",
            "returned_to_same_camp": "returned_to_same_camp",
            "missed_drop": "missed_drop_observed",
        }
        if kind in flag_events:
            self._flags[flag_events[kind]] = True

    @staticmethod
    def _count_map(raw: Any) -> Dict[str, int]:
        if not isinstance(raw, Mapping):
            return {}
        result: Dict[str, int] = {}
        for item, count in raw.items():
            try:
                result[normalize_name(item)] = int(count or 0)
            except (TypeError, ValueError):
                continue
        return result


class LongBenchEvaluator:
    def __init__(self, task: GoalTask, diagnostic: DiagnosticGraph):
        if task.task_id != diagnostic.task_id:
            raise ValueError("task and diagnostic task_id differ")
        self.task = task
        self.diagnostic = diagnostic
        self.reached: Set[str] = set()

    def update(self, step: int, state: StateSnapshot) -> EvaluationSnapshot:
        goal = self.task.evaluate(state)
        progress = self.diagnostic.evaluate(state, reached=self.reached)
        self.reached = set(progress.reached)
        return EvaluationSnapshot(
            task_id=self.task.task_id,
            step=step,
            goal_met=goal.met,
            goal_evidence=goal.evidence,
            diagnostic=progress,
        )
