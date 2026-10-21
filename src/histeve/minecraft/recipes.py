"""Recipes for Minecraft subtasks. Counts describe one recipe application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

PLANKS = tuple(
    f"{wood}_planks"
    for wood in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "crimson", "warped")
)


@dataclass(frozen=True)
class Ingredient:
    alternatives: tuple[str, ...]
    count: int

    def evidence(self, inventory: Mapping[str, int]) -> dict:
        known = {
            name: inventory[name]
            for name in self.alternatives
            if name in inventory and type(inventory[name]) is int and inventory[name] >= 0
        }
        total = sum(known.values())
        status = (
            "sufficient"
            if total >= self.count
            else "insufficient"
            if len(known) == len(self.alternatives)
            else "unknown"
        )
        return {
            "any_of_items": list(self.alternatives),
            "required_count": self.count,
            "visually_known_counts": known,
            "known_total_lower_bound": total,
            "status": status,
        }


@dataclass(frozen=True)
class CraftRecipe:
    output: str
    output_count: int
    station: str
    ingredients: tuple[Ingredient, ...]
    instruction: str

    def planning_contract(self, inventory: Mapping[str, int]) -> dict:
        evidence = [item.evidence(inventory) for item in self.ingredients]
        statuses = {item["status"] for item in evidence}
        readiness = (
            "insufficient"
            if "insufficient" in statuses
            else "unknown"
            if "unknown" in statuses
            else "sufficient"
        )
        return {
            "output": self.output,
            "output_per_recipe": self.output_count,
            "station": self.station,
            "ingredients": evidence,
            "inventory_materials": readiness,
            "instruction": self.instruction,
            "entry_condition": "Materials and the required crafting interface must be visually established. "
            "A station in inventory is not evidence of an open interface. "
            "Materials may already be on the cursor or in the grid when resuming; "
            "inventory readiness alone does not decide resumability.",
        }


RECIPES = {
    "oak_planks": CraftRecipe(
        "oak_planks",
        4,
        "inventory_2x2",
        (Ingredient(("oak_log",), 1),),
        "Open inventory, put one oak log in the 2x2 crafting grid, and take the four oak planks.",
    ),
    "birch_planks": CraftRecipe(
        "birch_planks",
        4,
        "inventory_2x2",
        (Ingredient(("birch_log",), 1),),
        "Open inventory, put one birch log in the 2x2 crafting grid, and take the four birch planks.",
    ),
    "crafting_table": CraftRecipe(
        "crafting_table",
        1,
        "inventory_2x2",
        (Ingredient(PLANKS, 4),),
        "Open inventory, fill the 2x2 crafting grid with four wooden planks, and take one crafting table.",
    ),
    "stick": CraftRecipe(
        "stick",
        4,
        "inventory_2x2",
        (Ingredient(PLANKS, 2),),
        "Open inventory, place two wooden planks vertically in one crafting-grid column, and take the sticks.",
    ),
    "torch": CraftRecipe(
        "torch",
        4,
        "inventory_2x2",
        (Ingredient(("coal", "charcoal"), 1), Ingredient(("stick",), 1)),
        "Open inventory, place coal or charcoal directly above a stick, and take the torches.",
    ),
    "wooden_pickaxe": CraftRecipe(
        "wooden_pickaxe",
        1,
        "crafting_table_3x3",
        (Ingredient(PLANKS, 3), Ingredient(("stick",), 2)),
        "Use the crafting table, place three wooden planks across the top row and two sticks down the middle column, and take one wooden pickaxe.",
    ),
    "stone_pickaxe": CraftRecipe(
        "stone_pickaxe",
        1,
        "crafting_table_3x3",
        (Ingredient(("cobblestone",), 3), Ingredient(("stick",), 2)),
        "Use the crafting table, place three cobblestone across the top row and two sticks down the middle column, and take one stone pickaxe.",
    ),
    "iron_pickaxe": CraftRecipe(
        "iron_pickaxe",
        1,
        "crafting_table_3x3",
        (Ingredient(("iron_ingot",), 3), Ingredient(("stick",), 2)),
        "Use the crafting table, place three iron ingots across the top row and two sticks down the middle column, and take one iron pickaxe.",
    ),
    "furnace": CraftRecipe(
        "furnace",
        1,
        "crafting_table_3x3",
        (Ingredient(("cobblestone",), 8),),
        "Use the crafting table, place eight cobblestone around the outside of the 3x3 grid, leave the center empty, and take one furnace.",
    ),
    "chest": CraftRecipe(
        "chest",
        1,
        "crafting_table_3x3",
        (Ingredient(PLANKS, 8),),
        "Use the crafting table, place eight wooden planks around the outside of the 3x3 grid, leave the center empty, and take one chest.",
    ),
}

ALIASES = {"planks": "oak_planks", "table": "crafting_table", "sticks": "stick", "torches": "torch"}

SMELTING_INPUTS = {
    "iron_ingot": ("iron_ore",),
    "charcoal": tuple(
        sorted(
            f"{prefix}{wood}_{kind}"
            for wood in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak")
            for prefix in ("", "stripped_")
            for kind in ("log", "wood")
        )
    ),
    "glass": ("red_sand", "sand"),
    "cooked_beef": ("beef",),
    "cooked_chicken": ("chicken",),
    "cooked_porkchop": ("porkchop",),
    "cooked_mutton": ("mutton",),
}


def validate_smelting_parameters(parameters: Mapping) -> None:
    source, output = parameters.get("input"), parameters.get("output")
    if (
        not isinstance(source, str)
        or not isinstance(output, str)
        or source not in SMELTING_INPUTS.get(output, ())
    ):
        raise ValueError("unsupported smelting input/output pair")


def smelting_knowledge(output_choices: list[str], inventory: Mapping[str, int]) -> dict:
    return {
        "scope": "shared_recipe_knowledge_not_observed_state_or_executable_actions",
        "game_contract": "minecraft_1_16_furnace",
        "recipes": {
            output: {
                "output": output,
                "output_per_recipe": 1,
                "station": "furnace",
                "input": Ingredient(SMELTING_INPUTS[output], 1).evidence(inventory),
            }
            for output in output_choices
        },
        "entry_condition": "Choose a valid input/output conversion and compatible fuel supported by visual evidence. "
        "A finished product is not its own smelting ingredient. A furnace in inventory is not an open interface. "
        "Materials may already be in the furnace; unknown inventory counts are not zero. "
        "Retrieve the output into inventory; an active flame alone is not completion.",
    }


def get_recipe(item: str) -> CraftRecipe:
    try:
        return RECIPES[ALIASES.get(item, item)]
    except KeyError as exc:
        raise ValueError(f"no deterministic recipe instruction for {item!r}") from exc


def crafting_knowledge(skill_id: str, item_choices: list[str], inventory: Mapping[str, int]) -> dict:
    if skill_id not in {"craft_inventory_item", "craft_table_item"}:
        return {}
    return {
        "scope": "shared_recipe_knowledge_not_observed_state_or_executable_actions",
        "recipes": {item: get_recipe(item).planning_contract(inventory) for item in item_choices},
    }
