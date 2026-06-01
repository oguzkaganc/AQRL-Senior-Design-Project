import argparse
import os
import sys
import time

import numpy as np
from adafruit_servokit import ServoKit


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.controllers.trot_planner import TrotPlanner
from src.robot.default_pose import DEFAULT_STANDING_QPOS_JOINTS


# Current physical robot neutral. Channel 8 is the replaced FR knee servo.
NEUTRAL_ANGLES = [60, 60, 60, 50, 80, 60, 90, 90, 140, 100, 90, 90]

# Physical robot standing calibration in servo-angle degrees.
# Order per leg: [joint0_lateral, joint1_hip, joint2_knee]
# Keep this table as the single source of truth for physical stand tuning.
STAND_CALIBRATION = {
    "FL": [0.0, 4.0, 0.0],
    "RL": [0.0, -6.0, 0.0],
    "FR": [0.0, 0.0, 0.0],
    "RR": [6.0, 6.0, 0.0],
}


def clamp(angle):
    return max(0, min(180, int(round(angle))))


def build_servo_kit():
    kit = ServoKit(channels=16)
    for servo_num in range(12):
        kit.servo[servo_num].actuation_range = 180
    return kit


def write_angles(kit, angles):
    for ch, angle in enumerate(angles):
        kit.servo[ch].angle = clamp(angle)


def qpos_to_servo_angles(qpos12):
    angles = [90.0 + 90.0 * float(q) for q in qpos12]

    for idx in [1, 2, 4, 5]:
        angles[idx] = 180.0 - angles[idx]

    offsets = [-30, -30, -30, -40, -10, -30, 0, 0, 50, 10, 0, 0]
    return [clamp(a + off) for a, off in zip(angles, offsets)]


def apply_stand_calibration(angles):
    biased = list(angles)
    leg_channels = {
        "FL": [0, 1, 2],
        "RL": [3, 4, 5],
        "FR": [6, 7, 8],
        "RR": [9, 10, 11],
    }
    for leg, deltas in STAND_CALIBRATION.items():
        channels = leg_channels[leg]
        for i, delta in enumerate(deltas):
            biased[channels[i]] += delta
    return [clamp(a) for a in biased]


def apply_front_lift_scale(qpos_targets, stand_qpos, front_lift_scale=1.0):
    scaled = np.array(qpos_targets, dtype=np.float32).copy()
    if abs(front_lift_scale - 1.0) < 1e-6:
        return scaled

    for joint_idx in [2, 8]:  # FL j3, FR j3
        offset = scaled[joint_idx] - stand_qpos[joint_idx]
        if offset > 0.0:
            scaled[joint_idx] = stand_qpos[joint_idx] + offset * front_lift_scale
    return scaled


def interpolate_pose(kit, start_angles, end_angles, steps, dt):
    for step in range(1, steps + 1):
        alpha = step / steps
        blended = [
            start + (end - start) * alpha
            for start, end in zip(start_angles, end_angles)
        ]
        write_angles(kit, blended)
        time.sleep(dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stand-seconds", type=float, default=1.5)
    parser.add_argument("--runtime-seconds", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.05, help="Robot command period")
    parser.add_argument("--step-length", type=float, default=0.035)
    parser.add_argument("--step-velocity", type=float, default=0.22)
    parser.add_argument("--clearance", type=float, default=0.020)
    parser.add_argument("--penetration", type=float, default=0.003)
    parser.add_argument("--ramp-seconds", type=float, default=1.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=4.0)
    parser.add_argument("--front-lift-scale", type=float, default=1.0)
    parser.add_argument("--return-neutral", action="store_true")
    args = parser.parse_args()

    kit = build_servo_kit()

    stand_qpos = DEFAULT_STANDING_QPOS_JOINTS.astype(np.float32)
    stand_angles = apply_stand_calibration(qpos_to_servo_angles(stand_qpos))

    max_joint_delta = None
    if args.max_joint_delta_deg > 0.0:
        max_joint_delta = np.deg2rad(args.max_joint_delta_deg)

    planner = TrotPlanner(dt=args.dt, Tswing=0.2, max_joint_delta_per_step=max_joint_delta)
    planner.reset(initial_targets=stand_qpos)

    print("Neutral:", NEUTRAL_ANGLES)
    print("Stand qpos:", np.round(stand_qpos, 4).tolist())
    print("Stand servo:", stand_angles)
    print(
        f"Trot params: L={args.step_length:.3f} vel={args.step_velocity:.3f} "
        f"cl={args.clearance:.3f} dt={args.dt:.3f}"
    )

    print("")
    print("Writing neutral pose...")
    write_angles(kit, NEUTRAL_ANGLES)
    time.sleep(1.0)

    print("Moving to standing pose...")
    interpolate_pose(kit, NEUTRAL_ANGLES, stand_angles, steps=25, dt=args.dt)
    time.sleep(args.stand_seconds)

    steps_total = max(1, int(args.runtime_seconds / args.dt))
    ramp_steps = max(1, int(args.ramp_seconds / args.dt))

    print(f"Starting open-loop trot... steps={steps_total}")
    current_angles = list(stand_angles)
    for step in range(steps_total):
        tick_start = time.time()

        qpos_targets = planner.compute_joint_targets(
            step_length=args.step_length,
            step_velocity=args.step_velocity,
            clearance_height=args.clearance,
            penetration_depth=args.penetration,
        )
        qpos_targets = apply_front_lift_scale(
            qpos_targets,
            stand_qpos,
            front_lift_scale=args.front_lift_scale,
        )
        alpha = min(1.0, (step + 1) / ramp_steps)
        blended_qpos = (1.0 - alpha) * stand_qpos + alpha * qpos_targets
        target_angles = apply_stand_calibration(qpos_to_servo_angles(blended_qpos))

        write_angles(kit, target_angles)
        current_angles = target_angles

        if step % max(1, int(0.5 / args.dt)) == 0:
            print(
                f"step={step:03d} alpha={alpha:.2f} "
                f"FL={target_angles[0:3]} RL={target_angles[3:6]} "
                f"FR={target_angles[6:9]} RR={target_angles[9:12]}"
            )

        elapsed = time.time() - tick_start
        sleep = args.dt - elapsed
        if sleep > 0:
            time.sleep(sleep)

    print("Trot bitti.")

    if args.return_neutral:
        print("Returning to neutral pose...")
        interpolate_pose(kit, current_angles, NEUTRAL_ANGLES, steps=25, dt=args.dt)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
