import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import ConstantSchedule

from src.envs.trot_imu_env import TrotImuEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--run-name", default="aqrl_ppo_run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--random-lateral-range", type=float, default=0.01)
    parser.add_argument("--random-yaw-range-deg", type=float, default=4.0)
    parser.add_argument("--control-dt", type=float, default=0.04)
    parser.add_argument("--base-step-length", type=float, default=0.03)
    parser.add_argument("--base-step-velocity", type=float, default=0.30)
    parser.add_argument("--base-clearance", type=float, default=0.040)
    parser.add_argument("--base-penetration", type=float, default=0.003)
    parser.add_argument("--target-forward-velocity", type=float, default=0.10)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--random-forward-velocity-range", type=float, default=0.0)
    parser.add_argument("--random-forward-velocity-min", type=float, default=0.0)
    parser.add_argument("--zero-forward-probability", type=float, default=0.0)
    parser.add_argument("--random-yaw-command-range", type=float, default=0.0)
    parser.add_argument("--random-yaw-command-min-abs", type=float, default=0.0)
    parser.add_argument("--straight-command-probability", type=float, default=1.0)
    parser.add_argument("--forward-velocity-reward-weight", type=float, default=1.5)
    parser.add_argument("--forward-velocity-error-scale", type=float, default=0.08)
    parser.add_argument("--yaw-rate-error-weight", type=float, default=3.0)
    parser.add_argument("--yaw-error-weight", type=float, default=2.5)
    parser.add_argument("--integrated-yaw-error-weight", type=float, default=1.5)
    parser.add_argument("--lateral-error-weight", type=float, default=0.8)
    parser.add_argument("--lateral-velocity-weight", type=float, default=0.8)
    parser.add_argument("--height-target", type=float, default=0.19)
    parser.add_argument("--height-error-weight", type=float, default=2.0)
    parser.add_argument("--yaw-residual-scale", type=float, default=0.020)
    parser.add_argument("--action-weight", type=float, default=0.08)
    parser.add_argument("--gait-action-weight", type=float, default=0.10)
    parser.add_argument("--yaw-action-weight", type=float, default=0.25)
    parser.add_argument("--lateral-body-action-weight", type=float, default=1.0)
    parser.add_argument("--lift-action-weight", type=float, default=0.15)
    parser.add_argument("--action-smoothness-weight", type=float, default=0.18)
    parser.add_argument("--action-saturation-weight", type=float, default=3.0)
    parser.add_argument("--max-lateral-error", type=float, default=0.0)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env_kwargs = {
        "render_mode": None,
        "randomize_reset": args.randomize_reset,
        "random_lateral_range": args.random_lateral_range,
        "random_yaw_range_deg": args.random_yaw_range_deg,
        "control_dt": args.control_dt,
        "base_step_length": args.base_step_length,
        "base_step_velocity": args.base_step_velocity,
        "base_clearance": args.base_clearance,
        "base_penetration": args.base_penetration,
        "target_forward_velocity": args.target_forward_velocity,
        "target_yaw_rate": args.target_yaw_rate,
        "random_forward_velocity_range": args.random_forward_velocity_range,
        "random_forward_velocity_min": args.random_forward_velocity_min,
        "zero_forward_probability": args.zero_forward_probability,
        "random_yaw_command_range": args.random_yaw_command_range,
        "random_yaw_command_min_abs": args.random_yaw_command_min_abs,
        "straight_command_probability": args.straight_command_probability,
        "forward_velocity_reward_weight": args.forward_velocity_reward_weight,
        "forward_velocity_error_scale": args.forward_velocity_error_scale,
        "yaw_rate_error_weight": args.yaw_rate_error_weight,
        "yaw_error_weight": args.yaw_error_weight,
        "integrated_yaw_error_weight": args.integrated_yaw_error_weight,
        "lateral_error_weight": args.lateral_error_weight,
        "lateral_velocity_weight": args.lateral_velocity_weight,
        "height_target": args.height_target,
        "height_error_weight": args.height_error_weight,
        "yaw_residual_scale": args.yaw_residual_scale,
        "action_weight": args.action_weight,
        "gait_action_weight": args.gait_action_weight,
        "yaw_action_weight": args.yaw_action_weight,
        "lateral_body_action_weight": args.lateral_body_action_weight,
        "lift_action_weight": args.lift_action_weight,
        "action_smoothness_weight": args.action_smoothness_weight,
        "action_saturation_weight": args.action_saturation_weight,
        "max_lateral_error": args.max_lateral_error,
    }

    env = Monitor(TrotImuEnv(**env_kwargs), filename=os.path.join(log_dir, "monitor.csv"))
    if args.check_env:
        check_env(TrotImuEnv(**env_kwargs), warn=True)

    if args.load_model is not None:
        model = PPO.load(
            args.load_model,
            env=env,
            device=args.device,
            verbose=1,
            tensorboard_log=log_dir,
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = ConstantSchedule(args.learning_rate)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device=args.device,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "net_arch": {
                    "pi": [256, 256],
                    "vf": [256, 256],
                },
            },
            tensorboard_log=log_dir,
        )

    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=model_dir,
        name_prefix=args.run_name,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint,
        tb_log_name=args.run_name,
        progress_bar=False,
    )
    final_path = os.path.join(model_dir, "final")
    model.save(final_path)
    env.close()
    print(f"saved: {final_path}.zip")


if __name__ == "__main__":
    main()
