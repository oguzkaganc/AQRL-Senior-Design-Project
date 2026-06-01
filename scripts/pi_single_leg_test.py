import argparse
import time

from adafruit_servokit import ServoKit


# Reference code ordering assumption:
# 0-2  -> FL
# 3-5  -> RL
# 6-8  -> FR
# 9-11 -> RR
LEG_CHANNELS = {
    "FL": [0, 1, 2],
    "RL": [3, 4, 5],
    "FR": [6, 7, 8],
    "RR": [9, 10, 11],
}

# Current physical robot neutral. Channel 8 is the replaced FR knee servo.
NEUTRAL_ANGLES = [60, 60, 60, 50, 80, 60, 90, 90, 140, 100, 90, 90]


def clamp(angle):
    return max(0, min(180, int(round(angle))))


def write_pose(kit, angles):
    for ch, angle in enumerate(angles):
        kit.servo[ch].angle = clamp(angle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", choices=["FL", "RL", "FR", "RR"], default="FL")
    parser.add_argument(
        "--joint",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="0/1/2 = servo index inside the selected leg.",
    )
    parser.add_argument("--delta", type=float, default=8.0, help="Small angle delta around neutral.")
    parser.add_argument("--hold", type=float, default=0.8, help="Hold time at each position.")
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    kit = ServoKit(channels=16)
    for servo_num in range(12):
        kit.servo[servo_num].actuation_range = 180

    angles = list(NEUTRAL_ANGLES)
    target_channel = LEG_CHANNELS[args.leg][args.joint]

    print("Writing neutral pose...")
    print("Leg:", args.leg, "joint:", args.joint, "channel:", target_channel)
    print("Neutral:", angles)
    write_pose(kit, angles)
    time.sleep(1.5)

    plus = list(angles)
    minus = list(angles)
    plus[target_channel] += args.delta
    minus[target_channel] -= args.delta

    print("Starting small motion test.")
    print("Plus :", [clamp(a) for a in plus])
    print("Minus:", [clamp(a) for a in minus])

    for i in range(args.cycles):
        print(f"cycle {i+1}/{args.cycles}: neutral")
        write_pose(kit, angles)
        time.sleep(args.hold)

        print(f"cycle {i+1}/{args.cycles}: plus")
        write_pose(kit, plus)
        time.sleep(args.hold)

        print(f"cycle {i+1}/{args.cycles}: neutral")
        write_pose(kit, angles)
        time.sleep(args.hold)

        print(f"cycle {i+1}/{args.cycles}: minus")
        write_pose(kit, minus)
        time.sleep(args.hold)

    print("Test finished, returning to neutral pose.")
    write_pose(kit, angles)
    time.sleep(1.0)


if __name__ == "__main__":
    main()
