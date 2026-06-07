ACTION_ID_TO_NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

ACTION_NAME_TO_ID = {
    name: idx for idx, name in ACTION_ID_TO_NAME.items()
}


def find_goal_pos(env):
    unwrapped = env.unwrapped

    if hasattr(unwrapped, "goal_pos"):
        return tuple(unwrapped.goal_pos)

    grid = unwrapped.grid

    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)

            if cell is not None and cell.type == "goal":
                return (x, y)

    raise RuntimeError("Goal position was not found.")


def get_desired_dir(agent_pos, goal_pos):
    """
    MiniGrid directions:
    0 = right
    1 = down
    2 = left
    3 = up
    """
    ax, ay = agent_pos
    gx, gy = goal_pos

    dx = gx - ax
    dy = gy - ay

    if dx > 0:
        return 0

    if dx < 0:
        return 2

    if dy > 0:
        return 1

    if dy < 0:
        return 3

    return None


def turn_towards(current_dir, desired_dir):
    if desired_dir is None:
        return ACTION_NAME_TO_ID["done"]

    diff = (desired_dir - current_dir) % 4

    if diff == 0:
        return ACTION_NAME_TO_ID["forward"]

    if diff == 1:
        return ACTION_NAME_TO_ID["right"]

    if diff == 3:
        return ACTION_NAME_TO_ID["left"]

    return ACTION_NAME_TO_ID["right"]


def expert_action(env):
    unwrapped = env.unwrapped

    agent_pos = tuple(unwrapped.agent_pos)
    agent_dir = int(unwrapped.agent_dir)
    goal_pos = find_goal_pos(env)

    desired_dir = get_desired_dir(agent_pos, goal_pos)
    action = turn_towards(agent_dir, desired_dir)

    return action


def expert_action_name(env):
    action_id = expert_action(env)
    return ACTION_ID_TO_NAME[action_id]