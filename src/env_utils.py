import numpy as np

from src.expert.shortest_path_expert import find_goal_pos


def randomize_agent(env, seed):
    rng = np.random.default_rng(seed)
    unwrapped = env.unwrapped
    grid = unwrapped.grid
    goal_pos = tuple(find_goal_pos(env))

    empty_positions = []

    for x in range(1, grid.width - 1):
        for y in range(1, grid.height - 1):
            if (x, y) == goal_pos:
                continue

            cell = grid.get(x, y)
            if cell is None:
                empty_positions.append((x, y))

    agent_pos = empty_positions[rng.integers(len(empty_positions))]
    agent_dir = int(rng.integers(4))

    unwrapped.agent_pos = agent_pos
    unwrapped.agent_dir = agent_dir