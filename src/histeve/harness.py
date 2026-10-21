"""The three-layer loop: plan, execute, verify, remember."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping

from .executor import Executor
from .goal import GoalTask
from .interfaces import (
    Command,
    CompletionDetector,
    Frame,
    InvalidPlannerResponse,
    InvalidVisualResponse,
    Planner,
    Policy,
    RuntimeConfig,
    VisualAssessment,
    VisualEnvironment,
    VisualVerifier,
    assessment_state,
    inventory_deltas,
    public_frame,
    validate_assessment,
)
from .memory import EpisodeRecord, Memory
from .state import StateSnapshot
from .subtasks import Subtask, Subtasks


@dataclass(frozen=True)
class RunResult:
    termination: str
    visual_goal_met: bool
    steps: int
    planner_calls: int
    detector_handoffs: int
    local_retries: int
    episodes: tuple[EpisodeRecord, ...]
    assessment: VisualAssessment
    policy_error: str = ""


class Harness:
    def __init__(
        self,
        registry: Subtasks,
        policy: Policy,
        planner: Planner,
        verifier: VisualVerifier,
        renderer: Callable[[Subtask, Mapping[str, Any]], str],
        memory: Memory,
        *,
        completion_criteria: Mapping[str, str] | None = None,
        detector: CompletionDetector | None = None,
        config: RuntimeConfig = RuntimeConfig(),
        clock: Callable[[], float] = time.monotonic,
    ):
        if config.handoff == "detector" and detector is None:
            raise ValueError("detector handoff requires a detector")
        self.registry, self.policy, self.planner = registry, policy, planner
        self.verifier, self.renderer, self.memory = verifier, renderer, memory
        self.detector, self.config, self.clock = detector, config, clock
        self.completion_criteria = (
            dict(completion_criteria)
            if completion_criteria is not None
            else {name: registry.get(name).completion_criterion for name in registry.ids()}
        )
        self._used = False

    def _assess(self, frames, command=None, before=None):
        value = self.verifier.assess(frames, command, before)
        references = list(before.observation_frames) if before else []
        if before:
            for landmark in before.landmark_frames.values():
                references.extend(landmark)
        value = validate_assessment(value, frames, references, require_goal=self._requires_goal)
        return self.memory.observe(value)

    def _visual_goal_met(self, task, assessment):
        if assessment.goal is not None:
            return assessment.goal.status == "complete"
        return bool(task.goal_predicates) and task.evaluate(assessment.state).met

    def _context(self, task, assessment, frames, eligible, requirements, steps_remaining):
        working = {
            "state": assessment_state(assessment),
            "last_outcome": assessment.status,
            "last_reason": assessment.reason,
            "requirements": requirements,
            "steps_remaining": steps_remaining,
        }
        if assessment.goal is not None:
            working["goal_assessment"] = asdict(assessment.goal)
        return {
            "goal": task.instruction,
            "frames": [public_frame(frame) for frame in frames],
            "working": working,
            "schemas": [
                item.planning_schema(assessment.state, self.completion_criteria[item.skill_id])
                for item in eligible
            ],
            "retrieved": self.memory.retrieve(
                task.instruction,
                state=assessment.state,
                task_id=task.task_id,
                failure_signature=assessment.reason if assessment.status != "success" else "",
            ),
            "world": dict(self.memory.world),
        }

    def _prepare(self, command, eligible, assessment, requirements):
        subtask = next((item for item in eligible if item.skill_id == command.skill_id), None)
        if subtask is None:
            raise InvalidPlannerResponse("planner selected an inapplicable subtask")
        try:
            parameters = subtask.normalize_parameters(command.parameters)
            subtask.validate_parameters(parameters)
        except ValueError as exc:
            raise InvalidPlannerResponse(str(exc)) from exc
        if "count" in parameters:
            item = parameters.get("output", parameters.get("item", ""))
            if item and parameters["count"] > 1:
                requirements[item] = {
                    "target": parameters["count"],
                    "remaining": max(0, parameters["count"] - assessment.state.inventory.get(item, 0)),
                }
            parameters["count"] = 1
        parameters = subtask.execution_parameters(parameters, self.memory.world)
        return subtask, Command(command.skill_id, parameters)

    def _record(self, task, subtask, command, before, after, start, end, requirements):
        gains = inventory_deltas(before, after)
        item = str(command.parameters.get("output", command.parameters.get("item", "")))
        if (
            after.status == "success"
            and item
            and ("count" in command.parameters or subtask.skill_id == "collect_drop")
            and gains.get(item, 0) <= 0
        ):
            after = replace(after, status="interrupted", reason="no_visual_inventory_gain")
        if after.status == "success" and subtask.family == "placement":
            old, new = before.state.placed.get(item), after.state.placed.get(item)
            if old is None or new is None or new <= old:
                after = replace(after, status="interrupted", reason="no_visual_placement_gain")
        for name, requirement in requirements.items():
            requirement["remaining"] = max(0, requirement["remaining"] - gains.get(name, 0))
        record = EpisodeRecord(
            episode_id=f"{self.memory.run_id}:{len(self.memory.current)}",
            task_id=task.task_id,
            seed=self.memory.seed,
            skill_id=subtask.skill_id,
            parameters=command.parameters,
            status=after.status,
            start_step=start,
            end_step=end,
            pre_state=assessment_state(before),
            post_state=assessment_state(after),
            progress_events=tuple(f"inventory_delta:{name}={count}" for name, count in gains.items()),
            failure_signature=after.reason or ("visual_failure" if after.status == "failure" else ""),
            evidence_refs=after.evidence_refs,
            context_tags=self.memory.state_tags(before.state),
        )
        self.memory.commit(record, task.instruction)
        return after

    def run(self, task: GoalTask, env: VisualEnvironment, initial: Frame) -> RunResult:
        """Run one episode from the initial observation."""
        if self._used or self.memory.current or self.memory.world:
            raise ValueError("create a fresh harness and memory for each episode")
        self._used = True
        self._requires_goal = not task.goal_predicates
        deadline = self.clock() + task.budget["minutes"] * 60
        step_limit = initial.step + task.budget["steps"]
        if task.budget["minutes"] <= 0 or task.budget["steps"] <= 0:
            raise ValueError("episode budgets must be positive")
        frames = (initial,)
        try:
            assessment = self._assess(frames)
        except InvalidVisualResponse as exc:
            return RunResult(
                "verifier_invalid_response",
                False,
                0,
                0,
                0,
                0,
                (),
                VisualAssessment(StateSnapshot(), ()),
                str(exc),
            )
        planner_calls = detector_handoffs = local_retries = retry_count = 0
        retry = None
        requirements = {}
        policy_error = ""
        executor = Executor(self.policy, self.detector, self.config, self.clock)

        while True:
            current = frames[-1]
            if self.clock() >= deadline:
                termination = "time_budget"
                break
            if self._visual_goal_met(task, assessment):
                termination = "visual_goal_candidate"
                break
            if current.terminal or current.step >= step_limit:
                termination = "environment_done" if current.terminal else "step_budget"
                break
            eligible = self.registry.applicable(assessment.state)
            if not eligible:
                termination = "no_applicable_skill"
                break

            try:
                if retry is not None and retry.skill_id in {item.skill_id for item in eligible}:
                    command = retry
                    local_retries += 1
                else:
                    retry_count = 0
                    context = self._context(
                        task, assessment, frames, eligible, requirements, step_limit - current.step
                    )
                    planner_calls += 1
                    command = self.planner.choose(copy.deepcopy(context))
                subtask, command = self._prepare(command, eligible, assessment, requirements)
            except InvalidPlannerResponse as exc:
                termination, policy_error = "planner_invalid_command", str(exc)
                break
            retry = None
            instruction = self.renderer(subtask, command.parameters)
            before = assessment
            execution = executor.run(
                env,
                frames,
                instruction,
                steps=min(subtask.default_budget_steps, step_limit - current.step),
                deadline=deadline,
            )
            frames = execution.frames
            detector_handoffs += execution.handoff == "detector"
            if execution.steps == 0 and execution.handoff == "time_budget":
                termination = "time_budget"
                break
            try:
                assessment = self._assess(frames, command, before)
            except InvalidVisualResponse as exc:
                termination, policy_error = "verifier_invalid_response", str(exc)
                break
            assessment = self._record(
                task, subtask, command, before, assessment, current.step, frames[-1].step, requirements
            )
            if (
                assessment.status != "success"
                and assessment.recoverable
                and retry_count < self.config.max_local_retries
            ):
                retry, retry_count = command, retry_count + 1
            else:
                retry_count = 0

        return RunResult(
            termination,
            self._visual_goal_met(task, assessment),
            frames[-1].step - initial.step,
            planner_calls,
            detector_handoffs,
            local_retries,
            tuple(self.memory.current),
            assessment,
            policy_error,
        )
