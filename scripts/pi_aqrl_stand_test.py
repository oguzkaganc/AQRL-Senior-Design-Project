import argparse
import time

from pi_servo_mapping import (
    NEUTRAL_ANGLES,
    build_servo_kit,
    mujoco_qpos_to_reference_servo_angles,
    write_angles,
)
from src.robot.default_pose import DEFAULT_STANDING_QPOS_JOINTS


STAND_CALIBRATION = {
    "FL": [0.0, 4.0, 0.0],
    "RL": [0.0, -6.0, 0.0],
    "FR": [0.0, 0.0, 0.0],
    "RR": [6.0, 6.0, 0.0],
}

LEG_CHANNELS = {
    "FL": [0, 1, 2],
    "RL": [3, 4, 5],
    "FR": [6, 7, 8],
    "RR": [9, 10, 11],
}


def apply_stand_calibration(angles):
    calibrated = list(angles)
    for leg, deltas in STAND_CALIBRATION.items():
        channels = LEG_CHANNELS[leg]
        for i, delta in enumerate(deltas):
            calibrated[channels[i]] += delta
    return [max(0, min(180, int(round(a)))) for a in calibrated]


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
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--use-runtime-calibration",
        action="store_true",
        help="Apply the same physical stand calibration used by open-loop/RL runtime.",
    )
    parser.add_argument("--return-neutral", action="store_true")
    args = parser.parse_args()

    kit = build_servo_kit()
    neutral = list(NEUTRAL_ANGLES)
    stand_qpos = list(DEFAULT_STANDING_QPOS_JOINTS)
    stand_angles = mujoco_qpos_to_reference_servo_angles(stand_qpos)
    if args.use_runtime_calibration:
        stand_angles = apply_stand_calibration(stand_angles)

    print("AQRL standing qpos:")
    print([round(v, 4) for v in stand_qpos])
    print("")
    print("Neutral servo angles:")
    print(neutral)
    print("")
    print("AQRL standing servo angles:")
    print(stand_angles)
    if args.use_runtime_calibration:
        print("")
        print("Runtime stand calibration applied:")
        print(STAND_CALIBRATION)

    print("")
    print("Writing neutral pose...")
    write_angles(kit, neutral)
    time.sleep(1.0)

    print("Moving to AQRL standing pose...")
    interpolate_pose(kit, neutral, stand_angles, steps=args.steps, dt=args.dt)
    print(f"Hold: {args.hold:.2f}s")
    time.sleep(args.hold)

    if args.return_neutral:
        print("Returning to neutral pose...")
        interpolate_pose(kit, stand_angles, neutral, steps=args.steps, dt=args.dt)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
