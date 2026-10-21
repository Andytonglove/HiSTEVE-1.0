"""Minecraft subtask definitions and recipe-aware planning."""

import json
from importlib.resources import files

from ..subtasks import Subtask, Subtasks
from .recipes import (
    ALIASES,
    crafting_knowledge,
    smelting_knowledge,
    validate_smelting_parameters,
)


class MinecraftSubtask(Subtask):
    def normalize_parameters(self, parameters):
        values = dict(parameters)
        if self.family in {"gui", "placement"} and "item" in values:
            values["item"] = ALIASES.get(values["item"], values["item"])
        if self.skill_id == "acquire_resource":
            values["item"] = {
                "tree": "oak_log",
                "trunk": "oak_log",
                "wood": "oak_log",
                "log": "oak_log",
                "stone": "cobblestone",
                "rock": "cobblestone",
                "ore": "iron_ore",
                "iron": "iron_ore",
                "coal_ore": "coal",
                "diamond_ore": "diamond",
                "beach": "sand",
                "riverbank": "sand",
                "desert": "sand",
            }.get(values.get("item"), values.get("item"))
        return values

    def execution_parameters(self, parameters, landmarks):
        values = dict(parameters)
        landmark = values.get("landmark_id")
        if self.skill_id == "return_to_landmark" and landmark in landmarks:
            detail = f"Match the recorded landmark: {landmarks[landmark]}"
            arrival = values.get("arrival_predicate", "")
            if detail not in arrival:
                values["arrival_predicate"] = f"{arrival}\n{detail}".strip()
        return values

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        if self.skill_id == "smelt_item":
            validate_smelting_parameters(parameters)

    def planning_schema(self, state, criterion):
        schema = super().planning_schema(state, criterion)
        properties = self.parameter_schema.get("properties", {})
        knowledge = crafting_knowledge(
            self.skill_id, properties.get("item", {}).get("enum", []), state.inventory
        )
        if knowledge:
            schema["crafting_knowledge"] = knowledge
        if self.skill_id == "smelt_item":
            schema["smelting_knowledge"] = smelting_knowledge(properties["output"]["enum"], state.inventory)
        return schema


def default_subtasks():
    rows = json.loads(files(__package__).joinpath("subtasks.json").read_text(encoding="utf-8"))["subtasks"]
    registry = Subtasks(MinecraftSubtask.from_dict(row) for row in rows)
    return registry, {name: registry.get(name).completion_criterion for name in registry.ids()}
