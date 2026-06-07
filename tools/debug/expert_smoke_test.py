from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from PIL import Image


# MiniGrid directions:
# 0 = right, 1 = down, 2 = left, 3 = up


def find_goal_pos(env):
    """
    Robust way to get goal position.
    In EmptyEnv there is usually env.unwrapped.goal_pos,
    but we also support scanning the grid.
    """
    unwrapped = env.unwrapped

    if hasattr(unwrapped, "goal_pos"):
        return tuple(unwrapped.goal_pos)

    grid = unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)

    raise RuntimeError("Goal position was not found")


def get_desired_dir(agent_pos, goal_pos):
    """
    Choose direction that moves agent closer to the goal.
    EmptyEnv has no obstacles, so Manhattan movement is enough.
    """
    ax, ay = agent_pos
    gx, gy = goal_pos

    dx = gx - ax
    dy = gy - ay

    if dx > 0:
        return 0  # right
    if dx < 0:
        return 2  # left
    if dy > 0:
        return 1  # down
    if dy < 0:
        return 3  # up

    return None  # already at goal


def turn_towards(current_dir, desired_dir, actions):
    """
    Return left/right/forward action to face desired direction.
    """
    if desired_dir is None:
        return actions.done

    diff = (desired_dir - current_dir) % 4

    if diff == 0:
        return actions.forward

    if diff == 1:
        return actions.right

    if diff == 3:
        return actions.left

    # Opposite direction: either left or right is fine.
    return actions.right


def expert_action(env):
    unwrapped = env.unwrapped

    agent_pos = tuple(unwrapped.agent_pos)
    agent_dir = int(unwrapped.agent_dir)
    goal_pos = find_goal_pos(env)

    desired_dir = get_desired_dir(agent_pos, goal_pos)
    action = turn_towards(agent_dir, desired_dir, unwrapped.actions)

    return action


def run_episode(env, seed, save_frames=False):
    obs, info = env.reset(seed=seed)

    frames = []
    total_reward = 0.0

    max_steps = env.unwrapped.max_steps

    for step in range(max_steps):
        if save_frames:
            frames.append(Image.fromarray(env.render()))

        action = expert_action(env)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            if save_frames:
                frames.append(Image.fromarray(env.render()))

            success = terminated and reward > 0
            return {
                "seed": seed,
                "success": success,
                "return": total_reward,
                "steps": step + 1,
                "terminated": terminated,
                "truncated": truncated,
            }, frames

    return {
        "seed": seed,
        "success": False,
        "return": total_reward,
        "steps": max_steps,
        "terminated": False,
        "truncated": True,
    }, frames


def main():
    output_dir = Path("outputs/expert_smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    env_id = "MiniGrid-Empty-8x8-v0"
    env = gym.make(env_id, render_mode="rgb_array")

    results = []

    for seed in range(20):
        result, frames = run_episode(
            env=env,
            seed=seed,
            save_frames=(seed == 0),
        )
        results.append(result)

        print(
            f"seed={result['seed']:02d} | "
            f"success={result['success']} | "
            f"return={result['return']:.3f} | "
            f"steps={result['steps']}"
        )

        if seed == 0 and frames:
            frames[0].save(
                output_dir / "expert_episode_seed0.gif",
                save_all=True,
                append_images=frames[1:],
                duration=300,
                loop=0,
            )

    env.close()

    success_rate = sum(r["success"] for r in results) / len(results)
    avg_return = sum(r["return"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    print()
    print("Summary")
    print(f"success_rate = {success_rate:.3f}")
    print(f"avg_return   = {avg_return:.3f}")
    print(f"avg_steps    = {avg_steps:.2f}")
    print(f"saved gif    = {output_dir / 'expert_episode_seed0.gif'}")


if __name__ == "__main__":
    main()