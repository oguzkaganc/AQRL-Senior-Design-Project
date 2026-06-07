import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from stable_baselines3 import PPO

from src.envs.trot_imu_env import TrotImuEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/aqrl_v9_1850k_selected/models/final.zip")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--control-dt", type=float, default=0.04)
    parser.add_argument("--base-step-length", type=float, default=0.03)
    parser.add_argument("--base-step-velocity", type=float, default=0.30)
    parser.add_argument("--base-clearance", type=float, default=0.040)
    parser.add_argument("--base-penetration", type=float, default=0.003)
    parser.add_argument("--yaw-residual-scale", type=float, default=0.026)
    parser.add_argument("--target-forward-velocity", type=float, default=0.10)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--random-forward-velocity-range", type=float, default=0.0)
    parser.add_argument("--random-yaw-command-range", type=float, default=0.0)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--random-lateral-range", type=float, default=0.01)
    parser.add_argument("--random-yaw-range-deg", type=float, default=4.0)
    parser.add_argument("--yaw-rate-error-weight", type=float, default=3.0)
    parser.add_argument("--yaw-error-weight", type=float, default=2.5)
    parser.add_argument("--integrated-yaw-error-weight", type=float, default=1.5)
    parser.add_argument("--lateral-error-weight", type=float, default=0.8)
    args = parser.parse_args()

    env = TrotImuEnv(
        render_mode=None if args.no_render else "human",
        control_dt=args.control_dt,
        base_step_length=args.base_step_length,
        base_step_velocity=args.base_step_velocity,
        base_clearance=args.base_clearance,
        base_penetration=args.base_penetration,
        yaw_residual_scale=args.yaw_residual_scale,
        target_forward_velocity=args.target_forward_velocity,
        target_yaw_rate=args.target_yaw_rate,
        random_forward_velocity_range=args.random_forward_velocity_range,
        random_yaw_command_range=args.random_yaw_command_range,
        randomize_reset=args.randomize_reset,
        random_lateral_range=args.random_lateral_range,
        random_yaw_range_deg=args.random_yaw_range_deg,
        yaw_rate_error_weight=args.yaw_rate_error_weight,
        yaw_error_weight=args.yaw_error_weight,
        integrated_yaw_error_weight=args.integrated_yaw_error_weight,
        lateral_error_weight=args.lateral_error_weight,
    )
    model = PPO.load(args.model, device="cpu")
    obs, _ = env.reset()

    total_reward = 0.0
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            print(
                f"step={step:4d} reward={reward:+.3f} "
                f"action={np.round(action, 3)} "
                f"pos=({env.data.qpos[0]:+.3f}, {env.data.qpos[1]:+.3f}, {env.data.qpos[2]:+.3f}) "
                f"imu=({obs[0]:+.3f}, {obs[1]:+.3f}, {obs[2]:+.3f}, {obs[3]:+.3f}) "
                f"yaw_err={info['yaw_error']:+.3f} "
                f"lat_err={info['lateral_error']:+.3f} "
                f"cmd=({info['target_forward_velocity']:+.3f}, {info['target_yaw_rate']:+.3f})"
            )

        if not args.no_render:
            time.sleep(env.control_dt)

        if terminated or truncated:
            print(
                "episode ended:",
                step,
                "terminated=",
                terminated,
                "truncated=",
                truncated,
            )
            obs, _ = env.reset()

    print("total_reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()
