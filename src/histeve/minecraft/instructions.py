"""Render semantic Minecraft requests as fixed instructions for the VLA."""

from typing import Any, Callable, Mapping

from ..subtasks import Subtask
from .recipes import get_recipe, validate_smelting_parameters

_ITEM_ALIASES = {
    "planks": "oak planks",
    "sticks": "sticks",
    "table": "crafting table",
    "wood": "oak log",
    "log": "oak log",
    "tree": "oak log",
    "ore": "iron ore",
    "iron": "iron ore",
    "rock": "stone",
}


class InstructionRenderer:
    """Deterministic semantic-schema to natural-language renderer."""

    GUI_SKILLS = frozenset(("craft_inventory_item", "craft_table_item", "smelt_item", "close_gui"))
    SEMANTIC_DETAILS = {
        "explore_for": ("strategy", "Exploration strategy"),
        "approach_target": ("stop_distance", "Stopping condition"),
        "acquire_resource": ("tool", "Use this tool"),
        "place_block": ("placement_constraint", "Placement constraint"),
        "hunt_entity": ("desired_drop", "Requested drop"),
        "record_landmark": ("description", "Landmark appearance"),
        "return_to_landmark": ("arrival_predicate", "Arrival condition"),
        "recover_navigation": ("failure_type", "Observed obstruction"),
    }

    @classmethod
    def _with_details(cls, instruction: str, skill_id: str, parameters: Mapping[str, Any]) -> str:
        key, label = cls.SEMANTIC_DETAILS[skill_id]
        detail = parameters.get(key, "")
        if not isinstance(detail, str):
            raise ValueError(f"{key} must be a semantic string")
        return f"{instruction}\n{label}: {detail.strip()}" if detail.strip() else instruction

    def __init__(
        self,
        *,
        route_callback: Callable[..., None] | None = None,
        command_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.route_callback, self.command_callback = route_callback, command_callback

    @staticmethod
    def _name(value: Any, *, item_aliases: bool = True) -> str:
        key = str(value or "target").strip().lower().replace("minecraft:", "").replace(" ", "_")
        return _ITEM_ALIASES.get(key, key.replace("_", " ")) if item_aliases else key.replace("_", " ")

    def __call__(self, skill: Subtask, parameters: Mapping[str, Any]) -> str:
        skill_id = skill.skill_id
        if skill_id == "smelt_item":
            validate_smelting_parameters(parameters)
        inspect_inventory = (
            skill_id == "observe_scene" and str(parameters.get("focus", "")).strip().lower() == "inventory"
        )
        is_gui = skill_id in self.GUI_SKILLS or inspect_inventory
        needs_table = skill_id == "craft_table_item"
        if self.route_callback:
            self.route_callback(is_gui=is_gui, needs_table=needs_table)
        if self.command_callback:
            self.command_callback(skill_id, parameters)

        target = self._name(
            parameters.get(
                "target", parameters.get("item", parameters.get("output", parameters.get("landmark_id")))
            ),
            item_aliases=skill_id
            not in {"explore_for", "approach_target", "record_landmark", "return_to_landmark"},
        )
        if skill_id in {"craft_inventory_item", "craft_table_item"}:
            key = str(parameters.get("item", "")).lower()
            return get_recipe(key).instruction
        if skill_id == "smelt_item":
            source = self._name(parameters.get("input"))
            fuel = str(parameters.get("fuel_policy", "available fuel")).replace("_", " ")
            return f"Use a furnace to smelt {source} with {fuel}, wait for completion, and take one {target}."
        if skill_id == "observe_scene":
            if inspect_inventory:
                return "Open your inventory and inspect its contents."
            return f"Look around carefully and visually inspect {self._name(parameters.get('focus', 'the scene'), item_aliases=False)}."
        if skill_id == "explore_for":
            return self._with_details(
                f"Explore the natural world until {target} is clearly visible.", skill_id, parameters
            )
        if skill_id == "approach_target":
            stopping = parameters.get("stop_distance", "")
            instruction = (
                f"Approach the visible {target} and keep it centered."
                if isinstance(stopping, str) and stopping.strip()
                else f"Approach the visible {target}, keep it centered, and stop when it is reachable."
            )
            return self._with_details(instruction, skill_id, parameters)
        if skill_id == "acquire_resource":
            target = {"cobblestone": "stone", "coal": "coal ore", "diamond": "diamond ore"}.get(
                target, target
            )
            return self._with_details(
                f"Find, approach, and mine {target}, then collect exactly one resulting item.",
                skill_id,
                parameters,
            )
        if skill_id == "collect_drop":
            return f"Locate and collect the nearby dropped {target}."
        if skill_id == "place_block":
            return self._with_details(
                f"Equip and place one {target} on a visible reachable surface.", skill_id, parameters
            )
        if skill_id == "hunt_entity":
            entity = self._name(parameters.get("entity"))
            return self._with_details(
                f"Find, approach, and defeat one {entity}, then collect its requested drop.",
                skill_id,
                parameters,
            )
        if skill_id == "record_landmark":
            return self._with_details(
                f"Visually inspect and remember the distinguishing appearance of this {target} landmark.",
                skill_id,
                parameters,
            )
        if skill_id == "return_to_landmark":
            return self._with_details(
                f"Navigate back to the remembered {target} using only visual landmark evidence.",
                skill_id,
                parameters,
            )
        if skill_id == "recover_navigation":
            return self._with_details(
                "Escape the visible obstruction, obtain a different usable viewpoint, and resume safe movement.",
                skill_id,
                parameters,
            )
        if skill_id == "recover_hazard":
            return f"Move away from the visible {self._name(parameters.get('hazard'))} hazard and reach visibly safe ground."
        if skill_id == "close_gui":
            return "Close the current GUI and return to first-person world control."
        raise ValueError(f"no instruction renderer for {skill_id}")
