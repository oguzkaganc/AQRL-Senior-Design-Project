import math

import numpy as np


ROLL_SCALE_RAD = 0.5
PITCH_SCALE_RAD = 0.5
YAW_RATE_SCALE_RAD_S = 2.0
YAW_ERROR_SCALE_RAD = 0.8
YAW_ERROR_LIMIT_RAD = 0.8


def normalize_imu_observation(
    roll_rad,
    pitch_rad,
    yaw_rate_rad_s,
    integrated_yaw_error_rad,
):
    """Return policy-facing IMU observation in MuJoCo env convention.

    The first two values are MuJoCo RPY components, not human labels:
    roll is rotation about MuJoCo X/right, pitch is rotation about MuJoCo
    Y/forward. Hardware drivers must transform their mounted sensor axes into
    this convention before calling this function.
    """
    values = np.array(
        [
            float(roll_rad) / ROLL_SCALE_RAD,
            float(pitch_rad) / PITCH_SCALE_RAD,
            float(yaw_rate_rad_s) / YAW_RATE_SCALE_RAD_S,
            float(integrated_yaw_error_rad) / YAW_ERROR_SCALE_RAD,
        ],
        dtype=np.float32,
    )
    return np.clip(values, -3.0, 3.0).astype(np.float32)


def deg_to_rad(deg):
    return float(deg) * math.pi / 180.0


def dps_to_rad_s(dps):
    return float(dps) * math.pi / 180.0
