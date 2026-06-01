import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.envs.trot_imu_env import TrotImuEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--target-forward-velocity", type=float, default=0.10)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--random-forward-velocity-range", type=float, default=0.0)
    parser.add_argument("--random-yaw-command-range", type=float, default=0.0)
    args = parser.parse_args()

    env = TrotImuEnv(
        render_mode="human" if args.render else None,
        target_forward_velocity=args.target_forward_velocity,
        target_yaw_rate=args.target_yaw_rate,
        random_forward_velocity_range=args.random_forward_velocity_range,
        random_yaw_command_range=args.random_yaw_command_range,
    )
    obs, _ = env.reset()
    rng = np.random.default_rng(123)
    total_reward = 0.0

    print("obs shape:", obs.shape)
    print("action shape:", env.action_space.shape)
    print("obs finite:", np.isfinite(obs).all())
    print("initial obs:", np.round(obs, 4))

    for step in range(args.steps):
        if args.random:
            action = rng.uniform(-0.25, 0.25, size=env.action_space.shape).astype(
                np.float32
            )
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            print(
                f"step={step:4d} reward={reward:+.3f} "
                f"pos=({env.data.qpos[0]:+.3f}, {env.data.qpos[1]:+.3f}, {env.data.qpos[2]:+.3f}) "
                f"imu=({obs[0]:+.3f}, {obs[1]:+.3f}, {obs[2]:+.3f}, {obs[3]:+.3f}) "
                f"cmd=({info['target_forward_velocity']:+.3f}, {info['target_yaw_rate']:+.3f}) "
                f"yaw_err={info['yaw_error']:+.3f} "
                f"lat_err={info['lateral_error']:+.3f} "
                f"step_length={info['step_length']:.3f} clearance={info['clearance']:.3f}"
            )

        if terminated or truncated:
            print("ended:", step, "terminated=", terminated, "truncated=", truncated)
            break

    print("total_reward:", total_reward)
    print("final_pos:", env.data.qpos[:3])
    env.close()


if __name__ == "__main__":
    main()
