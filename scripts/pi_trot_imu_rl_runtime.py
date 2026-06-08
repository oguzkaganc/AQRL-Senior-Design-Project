import argparse
import os
import sys
import time

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.controllers.trot_planner import TrotPlanner
from src.robot.default_pose import DEFAULT_STANDING_QPOS_JOINTS
from src.sensors.imu_observation import normalize_imu_observation


LEG_CHANNELS = {
    "FL": [0, 1, 2],
    "RL": [3, 4, 5],
    "FR": [6, 7, 8],
    "RR": [9, 10, 11],
}

# Current physical robot neutral. Channel 8 is the replaced FR knee servo.
NEUTRAL_ANGLES = [60, 60, 60, 50, 80, 60, 90, 90, 140, 100, 90, 90]

# MuJoCo qpos -> reference servo angle offsets. Channel 8 has +50 for the
# replaced FR knee servo neutral alignment.
SERVO_OFFSETS = [-30, -30, -30, -40, -10, -30, 0, 0, 50, 10, 0, 0]

# Physical standing calibration used by both open-loop and RL deployment tests.
STAND_CALIBRATION = {
    "FL": [0.0, 4.0, 0.0],
    "RL": [0.0, -6.0, 0.0],
    "FR": [0.0, 0.0, 0.0],
    "RR": [6.0, 6.0, 0.0],
}


def clamp(angle, angle_min=0.0, angle_max=180.0):
    return int(round(max(angle_min, min(angle_max, float(angle)))))


def build_servo_kit():
    from adafruit_servokit import ServoKit

    kit = ServoKit(channels=16)
    for servo_num in range(12):
        kit.servo[servo_num].actuation_range = 180
    return kit


def write_angles(kit, angles, angle_min=0.0, angle_max=180.0):
    for ch, angle in enumerate(angles):
        kit.servo[ch].angle = clamp(angle, angle_min, angle_max)


def qpos_to_servo_angles(qpos12, angle_min=0.0, angle_max=180.0):
    if len(qpos12) != 12:
        raise ValueError("Expected 12 qpos values")

    angles = [90.0 + 90.0 * float(q) for q in qpos12]
    for idx in [1, 2, 4, 5]:
        angles[idx] = 180.0 - angles[idx]

    return [
        clamp(angle + offset, angle_min, angle_max)
        for angle, offset in zip(angles, SERVO_OFFSETS)
    ]


def apply_stand_calibration(angles, enabled=True, angle_min=0.0, angle_max=180.0):
    if not enabled:
        return [clamp(a, angle_min, angle_max) for a in angles]

    biased = list(angles)
    for leg, deltas in STAND_CALIBRATION.items():
        channels = LEG_CHANNELS[leg]
        for i, delta in enumerate(deltas):
            biased[channels[i]] += delta
    return [clamp(a, angle_min, angle_max) for a in biased]


def limit_angle_step(target_angles, current_angles, max_step_deg):
    if max_step_deg <= 0.0:
        return list(target_angles)

    limited = []
    max_step = float(max_step_deg)
    for current, target in zip(current_angles, target_angles):
        delta = float(target) - float(current)
        delta = max(-max_step, min(max_step, delta))
        limited.append(float(current) + delta)
    return limited


def interpolate_pose(kit, start_angles, end_angles, steps, dt, angle_min, angle_max):
    for step in range(1, steps + 1):
        alpha = step / steps
        blended = [
            start + (end - start) * alpha
            for start, end in zip(start_angles, end_angles)
        ]
        write_angles(kit, blended, angle_min, angle_max)
        time.sleep(dt)


def phase_features(step_count, dt):
    phase = 2.0 * np.pi * step_count * dt / 0.4
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def build_obs(imu_obs, step_count, dt, prev_action, target_forward_velocity, target_yaw_rate):
    command = np.array([target_forward_velocity, target_yaw_rate], dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(imu_obs, dtype=np.float32),
            phase_features(step_count, dt),
            np.asarray(prev_action, dtype=np.float32),
            command,
        ]
    ).astype(np.float32)


def apply_deadband(value, deadband):
    value = float(value)
    deadband = max(0.0, float(deadband))
    magnitude = abs(value)
    if magnitude <= deadband:
        return 0.0
    return np.sign(value) * (magnitude - deadband)


def map_action(
    action,
    base_step_length,
    base_step_velocity,
    base_clearance,
    yaw_residual_scale,
):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return {
        "step_length": float(np.clip(base_step_length + 0.008 * action[0], 0.022, 0.040)),
        "step_velocity": float(np.clip(base_step_velocity + 0.04 * action[1], 0.24, 0.36)),
        "clearance": float(np.clip(base_clearance + 0.008 * action[2], 0.030, 0.052)),
        "lateral_fraction": float(0.006 * action[3]),
        "yaw_residual": float(yaw_residual_scale * action[4]),
        "body_y_offset": float(0.003 * action[5]),
        "lift_scales": np.clip(
            1.0 + 0.12 * np.asarray(action[6:10], dtype=np.float32),
            0.88,
            1.15,
        ).astype(np.float32),
    }


def apply_leg_lift_scales(
    qpos_targets,
    stand_qpos,
    lift_scales,
    front_lift_boost=1.0,
    rear_lift_boost=1.0,
):
    scaled = np.asarray(qpos_targets, dtype=np.float32).copy()
    knee_indices = [2, 8, 5, 11]  # FL, FR, RL, RR in policy action order.
    boosts = np.array(
        [front_lift_boost, front_lift_boost, rear_lift_boost, rear_lift_boost],
        dtype=np.float32,
    )
    for joint_idx, scale, boost in zip(knee_indices, lift_scales, boosts):
        scaled[joint_idx] = stand_qpos[joint_idx] + (
            scaled[joint_idx] - stand_qpos[joint_idx]
        ) * float(scale) * float(boost)
    return scaled


def load_actor(actor_path):
    if not os.path.isabs(actor_path):
        actor_path = os.path.join(REPO_ROOT, actor_path)
    actor = torch.jit.load(actor_path, map_location="cpu")
    actor.eval()
    return actor, actor_path


def build_imu(args):
    if args.no_imu:
        return None

    from src.sensors.mpu6050 import MPU6050

    imu = MPU6050(
        bus_id=args.imu_bus,
        alpha=args.imu_alpha,
        calibration_seconds=args.imu_calibration_seconds,
    )
    imu.open()
    return imu


def make_zero_imu_sample():
    return {
        "mujoco_roll_rad": 0.0,
        "mujoco_pitch_rad": 0.0,
        "mujoco_yaw_rate_rad_s": 0.0,
        "integrated_yaw_error_rad": 0.0,
        "stale_imu": False,
        "imu_read_error": "",
    }


def read_imu_obs(
    imu,
    fallback_obs,
    fallback_sample,
    retries,
    retry_delay,
):
    if imu is None:
        return np.zeros(4, dtype=np.float32), make_zero_imu_sample()

    last_error = None
    for _ in range(max(1, int(retries))):
        try:
            sample = imu.read()
            sample["stale_imu"] = False
            sample["imu_read_error"] = ""
            return sample["policy_imu_obs"], sample
        except OSError as exc:
            last_error = exc
            time.sleep(max(0.0, float(retry_delay)))

    if fallback_obs is None or fallback_sample is None:
        raise last_error

    sample = dict(fallback_sample)
    sample["stale_imu"] = True
    sample["imu_read_error"] = repr(last_error)
    return np.asarray(fallback_obs, dtype=np.float32), sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", default="runs/exported/aqrl_v9_1850k_actor_cpu.ts")
    parser.add_argument("--runtime-seconds", type=float, default=20.0)
    parser.add_argument("--stand-seconds", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--base-step-length", type=float, default=0.030)
    parser.add_argument("--base-step-velocity", type=float, default=0.30)
    parser.add_argument("--base-clearance", type=float, default=0.040)
    parser.add_argument("--base-penetration", type=float, default=0.003)
    parser.add_argument("--target-forward-velocity", type=float, default=0.10)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--yaw-residual-scale", type=float, default=0.026)
    parser.add_argument("--policy-scale", type=float, default=1.0)
    parser.add_argument(
        "--front-lift-boost",
        type=float,
        default=1.0,
        help="Extra multiplier applied only to FL/FR knee lift motion.",
    )
    parser.add_argument(
        "--rear-lift-boost",
        type=float,
        default=1.0,
        help="Extra multiplier applied only to RL/RR knee lift motion.",
    )
    parser.add_argument("--ramp-seconds", type=float, default=1.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=4.0)
    parser.add_argument("--max-servo-step-deg", type=float, default=4.0)
    parser.add_argument("--angle-min", type=float, default=0.0)
    parser.add_argument("--angle-max", type=float, default=180.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--imu-bus", type=int, default=1)
    parser.add_argument("--imu-alpha", type=float, default=0.98)
    parser.add_argument("--imu-calibration-seconds", type=float, default=3.0)
    parser.add_argument("--imu-read-retries", type=int, default=3)
    parser.add_argument("--imu-retry-delay", type=float, default=0.005)
    parser.add_argument("--max-consecutive-imu-errors", type=int, default=5)
    parser.add_argument(
        "--yaw-rate-deadband",
        type=float,
        default=0.06,
        help="Rad/s deadband before integrating yaw error for deploy IMU noise",
    )
    parser.add_argument(
        "--yaw-error-leak",
        type=float,
        default=0.05,
        help="1/s leak on gyro-integrated yaw error to limit long-term MPU6050 drift",
    )
    parser.add_argument("--no-imu", action="store_true", help="Use zero IMU obs")
    parser.add_argument("--no-servos", action="store_true", help="Do not write hardware servos")
    parser.add_argument("--no-stand-calibration", action="store_true")
    parser.add_argument("--return-neutral", action="store_true")
    args = parser.parse_args()

    actor, actor_path = load_actor(args.actor)
    kit = None if args.no_servos else build_servo_kit()
    imu = build_imu(args)

    stand_qpos = DEFAULT_STANDING_QPOS_JOINTS.astype(np.float32)
    stand_angles = apply_stand_calibration(
        qpos_to_servo_angles(stand_qpos, args.angle_min, args.angle_max),
        enabled=not args.no_stand_calibration,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
    )

    max_joint_delta = None
    if args.max_joint_delta_deg > 0.0:
        max_joint_delta = np.deg2rad(args.max_joint_delta_deg)
    planner = TrotPlanner(dt=args.dt, Tswing=0.2, max_joint_delta_per_step=max_joint_delta)
    planner.reset(initial_targets=stand_qpos)

    print("actor:", actor_path)
    print("obs_dim=18 action_dim=10")
    print(
        f"base L={args.base_step_length:.3f} vel={args.base_step_velocity:.3f} "
        f"cl={args.base_clearance:.3f} dt={args.dt:.3f}"
    )
    print(f"policy_scale={args.policy_scale:.2f} yaw_residual_scale={args.yaw_residual_scale:.3f}")
    print(f"front_lift_boost={args.front_lift_boost:.2f} rear_lift_boost={args.rear_lift_boost:.2f}")
    print("neutral:", NEUTRAL_ANGLES)
    print("stand_servo:", stand_angles)

    current_angles = list(NEUTRAL_ANGLES)
    if kit is not None:
        print("Writing neutral pose...")
        write_angles(kit, current_angles, args.angle_min, args.angle_max)
        time.sleep(1.0)

        print("Moving to standing pose...")
        interpolate_pose(
            kit,
            current_angles,
            stand_angles,
            steps=25,
            dt=args.dt,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
        )
        current_angles = list(stand_angles)
        time.sleep(args.stand_seconds)

    if imu is not None:
        print(f"Starting IMU calibration: keep the robot still for {args.imu_calibration_seconds:.1f}s.")
        biases = imu.calibrate()
        print("IMU bias:", {k: round(v, 4) for k, v in biases.items()})
        imu.integrated_yaw_error_rad = 0.0
        imu.prev_t = time.time()

    steps_total = max(1, int(args.runtime_seconds / args.dt))
    ramp_steps = max(1, int(args.ramp_seconds / args.dt))
    prev_action = np.zeros(10, dtype=np.float32)
    last_imu_obs = np.zeros(4, dtype=np.float32)
    last_imu_sample = make_zero_imu_sample()
    consecutive_imu_errors = 0
    runtime_yaw_error_rad = 0.0

    print(f"Starting IMU-RL trot... steps={steps_total}")
    try:
        for step in range(steps_total):
            tick_start = time.time()

            imu_obs, imu_sample = read_imu_obs(
                imu,
                fallback_obs=last_imu_obs,
                fallback_sample=last_imu_sample,
                retries=args.imu_read_retries,
                retry_delay=args.imu_retry_delay,
            )
            if imu_sample.get("stale_imu", False):
                consecutive_imu_errors += 1
                print(
                    "IMU read failed, using last sample "
                    f"({consecutive_imu_errors}/{args.max_consecutive_imu_errors}): "
                    f"{imu_sample.get('imu_read_error', '')}"
                )
                if consecutive_imu_errors > args.max_consecutive_imu_errors:
                    raise RuntimeError("Too many consecutive IMU read errors")
                imu_obs = last_imu_obs.copy()
            else:
                consecutive_imu_errors = 0
                raw_yaw_rate = float(imu_sample["mujoco_yaw_rate_rad_s"])
                policy_yaw_rate = apply_deadband(raw_yaw_rate, args.yaw_rate_deadband)
                imu_dt = float(imu_sample.get("dt", args.dt))
                imu_dt = float(np.clip(imu_dt, 1e-4, 0.2))
                runtime_yaw_error_rad += policy_yaw_rate * imu_dt
                leak = max(0.0, min(1.0, float(args.yaw_error_leak) * imu_dt))
                runtime_yaw_error_rad *= 1.0 - leak
                runtime_yaw_error_rad = float(np.clip(runtime_yaw_error_rad, -0.8, 0.8))

                imu_obs = normalize_imu_observation(
                    imu_sample["mujoco_roll_rad"],
                    imu_sample["mujoco_pitch_rad"],
                    policy_yaw_rate,
                    runtime_yaw_error_rad,
                )
                imu_sample["policy_yaw_rate_rad_s"] = policy_yaw_rate
                imu_sample["policy_integrated_yaw_error_rad"] = runtime_yaw_error_rad
                last_imu_obs = np.asarray(imu_obs, dtype=np.float32).copy()
                last_imu_sample = dict(imu_sample)

            obs = build_obs(
                imu_obs,
                step_count=step,
                dt=args.dt,
                prev_action=prev_action,
                target_forward_velocity=args.target_forward_velocity,
                target_yaw_rate=args.target_yaw_rate,
            )

            with torch.no_grad():
                raw_action = (
                    actor(torch.from_numpy(obs).unsqueeze(0))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            raw_action = np.clip(raw_action, -1.0, 1.0)
            scaled_raw_action = np.clip(args.policy_scale * raw_action, -1.0, 1.0)
            smoothed_action = 0.8 * prev_action + 0.2 * scaled_raw_action
            alpha = min(1.0, (step + 1) / ramp_steps)
            ramped_action = alpha * smoothed_action
            params = map_action(
                ramped_action,
                base_step_length=args.base_step_length,
                base_step_velocity=args.base_step_velocity,
                base_clearance=args.base_clearance,
                yaw_residual_scale=args.yaw_residual_scale,
            )
            commanded_yaw_rate = args.target_yaw_rate + params["yaw_residual"]

            qpos_targets = planner.compute_joint_targets(
                step_length=params["step_length"],
                step_velocity=params["step_velocity"],
                clearance_height=params["clearance"],
                penetration_depth=args.base_penetration,
                yaw_rate=commanded_yaw_rate,
                lateral_fraction=params["lateral_fraction"],
                body_pos_offset=[0.0, params["body_y_offset"], 0.0],
            )
            qpos_targets = apply_leg_lift_scales(
                qpos_targets,
                stand_qpos,
                params["lift_scales"],
                front_lift_boost=args.front_lift_boost,
                rear_lift_boost=args.rear_lift_boost,
            )

            target_angles = apply_stand_calibration(
                qpos_to_servo_angles(qpos_targets, args.angle_min, args.angle_max),
                enabled=not args.no_stand_calibration,
                angle_min=args.angle_min,
                angle_max=args.angle_max,
            )
            target_angles = limit_angle_step(
                target_angles,
                current_angles,
                max_step_deg=args.max_servo_step_deg,
            )

            if kit is not None:
                write_angles(kit, target_angles, args.angle_min, args.angle_max)
            current_angles = list(target_angles)
            prev_action = smoothed_action.astype(np.float32)

            if step % max(1, args.log_every) == 0:
                print(
                    f"step={step:03d} alpha={alpha:.2f} "
                    f"imu=[{imu_obs[0]:+.2f},{imu_obs[1]:+.2f},{imu_obs[2]:+.2f},{imu_obs[3]:+.2f}] "
                    f"yaw_raw={imu_sample['mujoco_yaw_rate_rad_s']:+.3f} "
                    f"yaw_rate={imu_sample.get('policy_yaw_rate_rad_s', imu_sample['mujoco_yaw_rate_rad_s']):+.3f} "
                    f"yaw_err={imu_sample.get('policy_integrated_yaw_error_rad', imu_sample['integrated_yaw_error_rad']):+.3f} "
                    f"a4={ramped_action[4]:+.3f} yaw_cmd={commanded_yaw_rate:+.3f} "
                    f"L={params['step_length']:.3f} cl={params['clearance']:.3f} "
                    f"lift={np.round(params['lift_scales'], 2).tolist()} "
                    f"boost=[{args.front_lift_boost:.2f},{args.front_lift_boost:.2f},{args.rear_lift_boost:.2f},{args.rear_lift_boost:.2f}]"
                )

            elapsed = time.time() - tick_start
            sleep = args.dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if args.return_neutral and kit is not None:
            print("Returning to neutral pose...")
            interpolate_pose(
                kit,
                current_angles,
                NEUTRAL_ANGLES,
                steps=25,
                dt=args.dt,
                angle_min=args.angle_min,
                angle_max=args.angle_max,
            )
            time.sleep(0.5)
        if imu is not None:
            imu.close()


if __name__ == "__main__":
    main()
