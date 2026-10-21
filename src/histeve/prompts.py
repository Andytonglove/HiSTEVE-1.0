"""Default prompts. Applications can pass replacements to the planner and verifier."""

PLANNER = """Choose the next applicable subtask from the offered schemas to advance the goal.
Use current images, working state and relevant memory. Return only skill_id and parameters.
Treat unknown possessions as unknown. Inspect or obtain materials when needed; when resuming
a GUI, consider materials already in its slots or on the cursor. Recipe knowledge describes
what is required, not what is currently possessed. A count is a desired inventory total;
the harness executes one acquisition or recipe at a time and tracks the remaining need.
Choose semantic objectives, without absolute coordinates or fixed camera orientations.
"""

VERIFIER = """Compare the supplied before and current images to assess state and subtask outcome.
Return success only when the completion condition is visually supported, failure for a visible
failure, and interrupted when neither can be established. Cite supplied image IDs, including
the latest current image. Previous state is a fallible belief; it is not fresh evidence.
Report visible object names, item counts, placed-block counts and boolean flags. Omit unknown
counts; use zero only when absence is established. For placement, compare the same scene.
Describe landmarks by appearance and relative surroundings, without absolute coordinates.
Use recorded landmark images to check identity when returning to a previously visited place.
"""

INVENTORY = """inventory_scope describes the latest current image:
storage_total: all backpack and hotbar slots are visible and readable;
hotbar_lower_bound: only readable hotbar stacks are counted;
unknown: inventory is not readable, so return an empty inventory object.
Exclude crafting/furnace slots, cursor items and world drops from inventory.
inventory_gains contains only newly acquired items established by before/after evidence,
such as a product visibly transferred from an output slot into storage; cite before and current
images for that transfer. A larger hotbar count,
moving an existing stack, or merely seeing an output is insufficient. If uncertain, return {}.
"""

GOAL = """Assess the public goal separately from the current subtask. Return complete only
when every requested product, count and historical condition is supported by supplied visual
evidence. Return incomplete for visible unmet requirements, uncertain for missing evidence.
This is the agent's stopping decision; external benchmark scoring is separate.
"""
