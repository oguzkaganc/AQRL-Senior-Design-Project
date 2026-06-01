"""
AQRL trot planner.

Pipeline:
  1. BezierGait            → foot trajectories in the gait-generator frame
  2. _body_ik              → hip-to-foot using SYMMETRIC kinematic hip anchors
                             and mirror-symmetric joint commands
  3. aqrl_leg_ik.solve     → MuJoCo joint angles (current XML convention)

Frames:
  Gait frame:  X=forward, Y=left,   Z=up
  MuJoCo base: X=right,   Y=forward, Z=up

Why symmetric kinematic hips:
  Real XML hips are asymmetric in X (front +0.105, rear -0.144). Using the
  real values produces asymmetric joint commands that physics amplifies
  (front lifts 2× rear). Using a symmetric kinematic model (midpoint
  ±0.1245) forces mirror-symmetric joint trajectories per diagonal pair
  → equal lift. The ~20mm X foot-placement error is absorbed by the stance
  phase implicitly (robot walks straight regardless).

Leg names:
  Robot order: [FL, RL, FR, RR]
  Gait order:  [FL, FR, BL, BR]
"""

from collections import OrderedDict

import numpy as np

from src.controllers.bezier_gait import BezierGait
from src.controllers.lie_algebra import RpToTrans
from src.controllers import aqrl_leg_ik as ik


LEG_ORDER = ["FL", "RL", "FR", "RR"]
GAIT_TO_ROBOT = {"FL": "FL", "FR": "FR", "BL": "RL", "BR": "RR"}

NOMINAL_HEIGHT = 0.19        # effective hip→foot reach = 0.21
                             # balance: low body + j2 away from saturation
NOMINAL_FOOT_Y_HALF = 0.09

# Kinematic hip anchors used BY THE IK (SYMMETRIC, midpoint of real front/rear X).
# Symmetric anchors → mirror-symmetric joint commands → equal lift, zero drift.
_HIP_X_SYM = 0.1245
HIP_ANCHORS_PY = {
    "FL": np.array([ _HIP_X_SYM,  0.037, 0.02]),
    "FR": np.array([ _HIP_X_SYM, -0.037, 0.02]),
    "BL": np.array([-_HIP_X_SYM,  0.037, 0.02]),
    "BR": np.array([-_HIP_X_SYM, -0.037, 0.02]),
}

# Foot NOMINAL positions in body frame (ASYMMETRIC, under the REAL XML hips).
# This decouples visual foot placement (under real hips) from IK kinematics
# (symmetric commands). h2f vector is foot_target - symmetric_anchor, so all
# legs receive the same h2f X component (-0.0195), keeping commands symmetric.
FOOT_NOMINAL_X = {"FL": 0.105, "FR": 0.105, "BL": -0.144, "BR": -0.144}

# Forward shift pulls thigh upright (j2 less tilted back) so feet plant
# directly under real hips. Y bias compensates body CoM right-side offset.
SUPPORT_SHIFT_PY = np.array([+0.025, 0.0, 0.0])

# Rear swing-lift amplification. Rear legs physically track less of the
# commanded lift than front (higher load at swing start, different mass
# distribution). Scale only the POSITIVE Z offset (lift) - stance
# penetration is left untouched.
REAR_LIFT_SCALE = 1.0


def _make_default_foot_transforms() -> "OrderedDict":
    height = NOMINAL_HEIGHT
    y_half = NOMINAL_FOOT_Y_HALF
    pf = OrderedDict()
    for key in ["FL", "FR", "BL", "BR"]:
        y_sign = +1.0 if key in ("FL", "BL") else -1.0
        p = np.array([FOOT_NOMINAL_X[key], y_sign * y_half, -height], dtype=float)
        p[0:2] += SUPPORT_SHIFT_PY[0:2]
        pf[key] = RpToTrans(np.eye(3), p)
    return pf


def _mujoco_rpy_to_gait_frame(body_rpy):
    return [float(body_rpy[1]), float(-body_rpy[0]), float(body_rpy[2])]


def _mujoco_pos_to_gait_frame(body_pos):
    return [float(body_pos[1]), float(-body_pos[0]), float(body_pos[2])]


def mujoco_quat_to_rpy(quat):
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


def _body_ik(gait_rpy, gait_pos, T_bf_py):
    """Custom body IK — body-frame gait, no rotation feedback (positive-feedback
    unstable in closed loop). Uses SYMMETRIC kinematic hip anchors."""
    del gait_rpy
    pos = np.array(gait_pos, dtype=float)
    h2f = {}
    for gait_leg, T in T_bf_py.items():
        p_bf = T[0:3, 3]
        hip_anchor = HIP_ANCHORS_PY[gait_leg]
        h2f[gait_leg] = p_bf - pos - hip_anchor
    return h2f


class TrotPlanner:
    def __init__(self, dt=0.02, Tswing=0.2, max_joint_delta_per_step=None):
        self.bezier = BezierGait(dt=dt, Tswing=Tswing)
        self.dt = dt
        self.T_bf_default = _make_default_foot_transforms()
        self.max_joint_delta_per_step = max_joint_delta_per_step
        self.prev_targets = None

    def reset(self, initial_targets=None):
        self.bezier.reset()
        if initial_targets is None:
            self.prev_targets = None
        else:
            self.prev_targets = np.asarray(initial_targets, dtype=np.float32).copy()

    def compute_joint_targets(
        self,
        step_length=0.03,
        step_velocity=0.3,
        clearance_height=0.04,
        penetration_depth=0.005,
        lateral_fraction=0.0,
        yaw_rate=0.0,
        body_rpy=None,
        body_pos_offset=None,
    ):
        gait_rpy = _mujoco_rpy_to_gait_frame(body_rpy) if body_rpy is not None else [0.0, 0.0, 0.0]
        gait_pos = _mujoco_pos_to_gait_frame(body_pos_offset) if body_pos_offset is not None else [0.0, 0.0, 0.0]

        T_bf = self.bezier.GenerateTrajectory(
            L=step_length,
            LateralFraction=lateral_fraction,
            YawRate=yaw_rate,
            vel=step_velocity,
            T_bf_=self.T_bf_default,
            clearance_height=clearance_height,
            penetration_depth=penetration_depth,
            dt=self.dt,
        )

        # Amplify rear swing lift to compensate for worse PD tracking.
        for key in ("BL", "BR"):
            nominal_z = self.T_bf_default[key][2, 3]
            offset = T_bf[key][2, 3] - nominal_z
            if offset > 0.0:
                T_bf[key][2, 3] = nominal_z + offset * REAR_LIFT_SCALE

        hip_to_foot = _body_ik(gait_rpy, gait_pos, T_bf)

        mj_angles = np.zeros(12)
        for gait_leg in ["FL", "FR", "BL", "BR"]:
            robot_leg = GAIT_TO_ROBOT[gait_leg]
            h2f_py = hip_to_foot[gait_leg]
            r = ik.solve_from_gait_frame(h2f_py, robot_leg, prefer_branch="neg")
            robot_idx = LEG_ORDER.index(robot_leg)
            mj_angles[robot_idx * 3:(robot_idx + 1) * 3] = r.angles

        targets = mj_angles.astype(np.float32)
        if self.max_joint_delta_per_step is not None:
            max_delta = float(self.max_joint_delta_per_step)
            if max_delta > 0.0 and self.prev_targets is not None:
                delta = np.clip(targets - self.prev_targets, -max_delta, max_delta)
                targets = (self.prev_targets + delta).astype(np.float32)
            self.prev_targets = targets.copy()

        return targets


_planner = None
_planner_dt = None


def reset_planner():
    """Reset the module-level convenience planner, if it exists."""
    if _planner is not None:
        _planner.reset()


def compute_joint_targets(
    step_count: int = 0,
    frequency: float = 1.0,
    step_length: float = 0.03,
    step_height: float = 0.015,
    body_height: float = NOMINAL_HEIGHT,
    dt: float = 0.02,
    left_step_scale: float = 1.0,
    right_step_scale: float = 1.0,
    lateral_fraction: float = 0.0,
    yaw_rate: float = 0.0,
    penetration_depth: float = 0.003,
    body_rpy=None,
    body_pos=None,
    body_pos_offset=None,
    **kwargs,
) -> np.ndarray:
    del step_count, left_step_scale, right_step_scale, body_pos, kwargs

    global _planner, _planner_dt
    if _planner is None or _planner_dt != dt:
        _planner = TrotPlanner(dt=dt)
        _planner_dt = dt

    step_velocity = max(0.05, step_length * 2.0 * frequency)
    clearance_height = float(np.clip(step_height, 0.005, 0.06))

    if body_pos_offset is None and body_height is not None:
        body_pos_offset = [0.0, 0.0, float(body_height) - NOMINAL_HEIGHT]

    return _planner.compute_joint_targets(
        step_length=step_length / 2.0,
        step_velocity=step_velocity,
        clearance_height=clearance_height,
        penetration_depth=penetration_depth,
        lateral_fraction=lateral_fraction,
        yaw_rate=yaw_rate,
        body_rpy=body_rpy,
        body_pos_offset=body_pos_offset,
    )
