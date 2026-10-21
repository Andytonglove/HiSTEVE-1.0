"""Bounded VLA execution and completion-detector handoff."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .interfaces import DetectorPrediction, Frame, NativeAction, RuntimeConfig


class VLA:
    """Bind a model's native-action decoder to one fixed subtask instruction."""

    def __init__(self, infer: Callable, *, reset: Callable | None = None, route: Callable | None = None):
        self.infer, self.reset_model, self.route_model = infer, reset, route
        self.instruction = ""

    def set_route(self, *, is_gui: bool, needs_table: bool = False):
        if self.route_model is not None:
            self.route_model(is_gui=is_gui, needs_table=needs_table)

    def begin(self, instruction: str):
        if not instruction.strip():
            raise ValueError("instruction cannot be empty")
        self.instruction = instruction
        if self.reset_model is not None:
            self.reset_model()

    def act(self, frame: Frame, instruction: str) -> NativeAction:
        if instruction != self.instruction:
            raise ValueError("instruction changed inside a bounded invocation")
        action = self.infer(frame, instruction)
        if not isinstance(action, NativeAction):
            raise TypeError("the policy adapter must decode tokens to NativeAction")
        return action


class CompletionDetector:
    """Query four historical frames and the policy prefix before action decoding."""

    def __init__(self, query: Callable, policy_context: Callable, *, stride: int = 4):
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        self.query, self.policy_context, self.stride = query, policy_context, stride
        self.frames = deque(maxlen=4 * stride + 1)
        self.instruction = ""
        self.query_id = 0

    def reset(self, instruction: str):
        self.frames.clear()
        self.instruction = instruction

    def predict(self, frame: Frame, instruction: str) -> DetectorPrediction | None:
        if instruction != self.instruction:
            raise ValueError("detector instruction changed inside an invocation")
        if self.frames:
            if frame == self.frames[-1]:
                return None
            if frame.step != self.frames[-1].step + 1:
                raise ValueError("detector requires consecutive observations")
        self.frames.append(frame)
        if len(self.frames) < self.frames.maxlen:
            return None
        history = tuple(self.frames[i] for i in range(0, 4 * self.stride, self.stride))
        context = self.policy_context(frame, instruction)
        probability = float(self.query(history, frame, instruction, context))
        self.query_id += 1
        return DetectorPrediction(self.query_id, probability)


class HandoffGate:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.positives = 0
        self.last_query_id: int | None = None

    def update(self, probability: float | None, query_id: int | None = None) -> bool:
        if query_id is not None:
            if type(query_id) is not int or query_id < 0:
                raise ValueError("query_id must be a nonnegative integer")
            if self.last_query_id is not None and query_id < self.last_query_id:
                raise ValueError("detector query IDs went backwards")
            if query_id == self.last_query_id:
                return False
            self.last_query_id = query_id
        if probability is not None and (not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("invalid detector probability")
        # Missing predictions reset the streak.
        self.positives = self.positives + 1 if probability is not None and probability > self.threshold else 0
        return self.positives >= 2


@dataclass(frozen=True)
class Execution:
    frames: tuple[Frame, ...]
    steps: int
    handoff: str


class Executor:
    """Run one fixed instruction until handoff, budget, or environment termination."""

    def __init__(self, policy, detector, config: RuntimeConfig, clock):
        self.policy, self.detector = policy, detector
        self.config, self.clock = config, clock

    def run(self, env, frames, instruction, *, steps, deadline) -> Execution:
        history = deque(frames, maxlen=self.config.history_frames)
        start = history[-1].step
        self.policy.begin(instruction)
        gate = HandoffGate(self.config.threshold)
        if self.config.handoff == "detector":
            self.detector.reset(instruction)
        handoff = "budget"
        for _ in range(steps):
            if self.clock() >= deadline:
                handoff = "time_budget"
                break
            if self.config.handoff == "detector":
                prediction = self.detector.predict(history[-1], instruction)
                if prediction is not None and gate.update(prediction.probability, prediction.query_id):
                    handoff = "detector"
                    break
                if self.clock() >= deadline:
                    handoff = "time_budget"
                    break
            action = self.policy.act(history[-1], instruction)
            if not isinstance(action, NativeAction):
                raise TypeError("policy must return NativeAction")
            if self.clock() >= deadline:
                handoff = "time_budget"
                break
            frame = env.step(action)
            if frame.step != history[-1].step + 1:
                raise ValueError("one native action must advance one environment step")
            history.append(frame)
            if frame.terminal:
                handoff = "environment_done"
                break
        return Execution(tuple(history), history[-1].step - start, handoff)
