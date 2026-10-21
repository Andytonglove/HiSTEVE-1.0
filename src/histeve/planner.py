"""Image-based semantic planning and outcome verification."""

from __future__ import annotations

import json
from dataclasses import asdict

from . import prompts
from .interfaces import (
    Command,
    InvalidPlannerResponse,
    InvalidVisualResponse,
    VisualAssessment,
    VisualGoalAssessment,
    public_frame,
    validate_assessment,
)
from .state import StateSnapshot
from .subtasks import Subtask


class JsonPlanner:
    def __init__(self, query, *, prompt=prompts.PLANNER):
        self.query, self.prompt = query, prompt

    def choose(self, context):
        images = [row["image_ref"] for row in context["frames"]]
        request = {
            **context,
            "operation": self.prompt,
            "response_schema": {"skill_id": "string", "parameters": {}},
        }
        try:
            result = self.query(images, request)
            if (
                not isinstance(result, dict)
                or set(result) != {"skill_id", "parameters"}
                or not isinstance(result["skill_id"], str)
                or not isinstance(result["parameters"], dict)
            ):
                raise InvalidPlannerResponse("return only string skill_id and object parameters")
            offered = context.get("schemas", [])
            if offered:
                schema = next((row for row in offered if row["skill_id"] == result["skill_id"]), None)
                if schema is None:
                    raise InvalidPlannerResponse("planner selected an inapplicable subtask")
                Subtask(result["skill_id"], "", "", schema["parameter_schema"]).validate_parameters(
                    result["parameters"]
                )
            return Command(result["skill_id"], result["parameters"])
        except (json.JSONDecodeError, ValueError) as exc:
            raise InvalidPlannerResponse(str(exc)) from exc


class JsonVisualVerifier:
    def __init__(self, query, criteria, *, public_goal=None, prompt=prompts.VERIFIER):
        self.query, self.criteria, self.public_goal = query, dict(criteria), public_goal
        self.prompt = prompt

    def assess(self, frames, command, before):
        before_frames = before.observation_frames if before else ()
        landmark_frames = (
            [frame for group in before.landmark_frames.values() for frame in group] if before else []
        )
        images = list(dict.fromkeys(frame.image_ref for frame in (*landmark_frames, *before_frames, *frames)))
        aliases = {path: f"image_{index}" for index, path in enumerate(images)}
        paths = {alias: path for path, alias in aliases.items()}

        def describe(frame):
            return {**public_frame(frame), "image_ref": aliases[frame.image_ref]}

        request = {
            "operation": self.prompt,
            "before_frames": [describe(frame) for frame in before_frames],
            "frames": [describe(frame) for frame in frames],
            "image_order": list(paths),
            "command": asdict(command) if command else None,
            "completion_criterion": self.criteria[command.skill_id]
            if command
            else "Describe the initial state.",
            "previous_visual_state": asdict(before.state) if before else None,
            "landmarks": {
                name: {
                    "description": description,
                    "evidence_refs": [
                        aliases[frame.image_ref] for frame in before.landmark_frames.get(name, ())
                    ],
                }
                for name, description in before.landmarks.items()
            }
            if before
            else {},
            "inventory_observation_contract": prompts.INVENTORY,
            "response_schema": {
                "state": {"inventory": {}, "visible": [], "placed": {}, "flags": {}},
                "inventory_scope": "storage_total|hotbar_lower_bound|unknown",
                "inventory_gains": {},
                "evidence_refs": [aliases[frames[-1].image_ref]],
                "status": "success|failure|interrupted",
                "reason": "string",
                "recoverable": False,
                "landmarks": {},
            },
        }
        if self.public_goal is not None:
            request.update(public_goal=self.public_goal, goal_verification=prompts.GOAL)
            request["response_schema"]["goal_assessment"] = {
                "status": "complete|incomplete|uncertain",
                "reason": "string",
                "evidence_refs": [aliases[frames[-1].image_ref]],
            }
        try:
            response = self.query(images, request)
            value = self._parse(response, paths)
            return validate_assessment(
                value,
                frames,
                (*landmark_frames, *before_frames),
                require_goal=self.public_goal is not None,
            )
        except (json.JSONDecodeError, InvalidVisualResponse) as exc:
            raise InvalidVisualResponse(str(exc)) from exc

    def _parse(self, result, paths):
        fields = {
            "state",
            "inventory_scope",
            "inventory_gains",
            "evidence_refs",
            "status",
            "reason",
            "recoverable",
            "landmarks",
        }
        if self.public_goal is not None:
            fields.add("goal_assessment")
        if not isinstance(result, dict) or set(result) != fields or not isinstance(result["state"], dict):
            raise InvalidVisualResponse("invalid visual verification response schema")
        state = result["state"]
        if set(state) - {"inventory", "visible", "placed", "flags"}:
            raise InvalidVisualResponse("visual state includes unsupported fields")
        for counts in (state.get("inventory", {}), state.get("placed", {}), result["inventory_gains"]):
            if not isinstance(counts, dict) or any(
                type(value) is not int or value < 0 for value in counts.values()
            ):
                raise InvalidVisualResponse("visual counts must be nonnegative integers")
        visible, flags = state.get("visible", []), state.get("flags", {})
        if not isinstance(visible, list) or not all(isinstance(item, str) for item in visible):
            raise InvalidVisualResponse("visible must be a list of object names")
        if not isinstance(flags, dict) or any(type(value) is not bool for value in flags.values()):
            raise InvalidVisualResponse("flags must be booleans")
        if type(result["recoverable"]) is not bool or not isinstance(result["reason"], str):
            raise InvalidVisualResponse("invalid outcome fields")

        def references(raw):
            if not isinstance(raw, list) or not all(isinstance(ref, str) and ref in paths for ref in raw):
                raise InvalidVisualResponse("cite only supplied image IDs")
            return tuple(paths[ref] for ref in raw)

        goal = None
        if self.public_goal is not None:
            raw = result["goal_assessment"]
            if not isinstance(raw, dict) or set(raw) != {"status", "reason", "evidence_refs"}:
                raise InvalidVisualResponse("invalid goal assessment")
            if not isinstance(raw["reason"], str):
                raise InvalidVisualResponse("goal reason must be a string")
            goal = VisualGoalAssessment(raw["status"], raw["reason"], references(raw["evidence_refs"]))
        return VisualAssessment(
            StateSnapshot.from_mapping(state),
            references(result["evidence_refs"]),
            result["status"],
            result["reason"],
            result["recoverable"],
            result["landmarks"],
            goal=goal,
            inventory_scope=result["inventory_scope"],
            inventory_gains=dict(result["inventory_gains"]),
        )
