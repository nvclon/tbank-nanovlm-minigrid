ACTION_ONLY_PROMPT = """You are controlling an agent in MiniGrid.
The goal is to reach the green square.
Valid actions: left, right, forward.
Return exactly one action."""

TEXT_ACTION_PROMPT = """You are controlling an agent in MiniGrid.
Describe the situation briefly and choose the next action.
Use exactly this format:
Observation: <one short sentence>
Plan: <one short sentence>
Action: <left/right/forward>"""

ZERO_SHOT_ACTION_PROMPT = """You are controlling a MiniGrid agent from a top-down image.

Objects:
- The red triangle is the agent.
- The tip of the red triangle shows the direction the agent is facing.
- The green square is the goal.

Available actions:
- left: rotate the agent 90 degrees left
- right: rotate the agent 90 degrees right
- forward: move one cell in the direction the red triangle is facing

Task:
Choose the single best next action to reach the green goal.

Important:
- If the agent is already facing toward the goal, choose forward.
- If the agent is not facing toward the goal, choose left or right to turn toward it.
- Do not explain.
- Do not write a sentence.
- Output exactly one token from this list: left, right, forward.

Answer:"""

def make_action_only_target(action_name: str) -> str:
    return action_name

def make_text_action_target(action_name: str, agent_pos, goal_pos, agent_dir) -> str:
    ax, ay = agent_pos
    gx, gy = goal_pos

    if action_name == "forward":
        observation = "The agent is facing a useful direction toward the goal."
        plan = "Move forward to get closer to the green goal square."
    elif action_name == "left":
        observation = "The goal is not directly in front of the agent."
        plan = "Turn left to face a better direction."
    elif action_name == "right":
        observation = "The goal is not directly in front of the agent."
        plan = "Turn right to face a better direction."
    else:
        observation = "The agent is already at the goal."
        plan = "Finish the episode."

    return f"Observation: {observation}\nPlan: {plan}\nAction: {action_name}"