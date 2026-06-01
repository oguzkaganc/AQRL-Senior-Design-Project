from adafruit_servokit import ServoKit


LEG_CHANNELS = {
    "FL": [0, 1, 2],
    "RL": [3, 4, 5],
    "FR": [6, 7, 8],
    "RR": [9, 10, 11],
}

LEG_ORDER = ["FL", "RL", "FR", "RR"]

# Current physical robot neutral. Channel 8 is the replaced FR knee servo.
NEUTRAL_ANGLES = [60, 60, 60, 50, 80, 60, 90, 90, 140, 100, 90, 90]

# Physical directions confirmed on the real robot.
# +1 means positive delta increases the angle in the named physical direction.
# -1 means positive delta moves in the opposite physical direction.
#
# joint 0: lateral, positive -> robot's right
# joint 1: fore/aft, positive -> rear on left legs, front on right legs
# joint 2: knee, positive -> down on left legs, up on right legs
JOINT_SIGNS = {
    "FL": [1, 1, 1],
    "RL": [1, 1, 1],
    "FR": [1, 1, 1],
    "RR": [1, 1, 1],
}


def clamp(angle):
    return max(0, min(180, int(round(angle))))


def build_servo_kit(channels=16):
    kit = ServoKit(channels=channels)
    for servo_num in range(12):
        kit.servo[servo_num].actuation_range = 180
    return kit


def write_angles(kit, angles):
    for ch, angle in enumerate(angles):
        kit.servo[ch].angle = clamp(angle)


def apply_leg_deltas(base_angles, leg_deltas_deg):
    angles = list(base_angles)
    for leg, deltas in leg_deltas_deg.items():
        if leg not in LEG_CHANNELS:
            raise ValueError(f"Unknown leg: {leg}")
        if len(deltas) != 3:
            raise ValueError(f"{leg} delta list must have 3 values")
        channels = LEG_CHANNELS[leg]
        signs = JOINT_SIGNS[leg]
        for joint, delta in enumerate(deltas):
            angles[channels[joint]] += signs[joint] * delta
    return [clamp(a) for a in angles]


def mujoco_qpos_to_reference_servo_angles(qpos12):
    """
    Convert 12 MuJoCo joint targets in our order [FL, RL, FR, RR]
    to the raw servo angles used on Raspberry Pi.

    This mapping matches the calibrated physical servo convention:
      action_to_send = q * 90 + 90
      invert channels 1:3 and 4:6
      then apply per-channel offsets
    """
    if len(qpos12) != 12:
        raise ValueError("Expected 12 joint values")

    angles = [90.0 + 90.0 * float(q) for q in qpos12]

    for idx in [1, 2, 4, 5]:
        angles[idx] = 180.0 - angles[idx]

    offsets = [-30, -30, -30, -40, -10, -30, 0, 0, 50, 10, 0, 0]
    angles = [angle + offset for angle, offset in zip(angles, offsets)]
    return [clamp(a) for a in angles]
