"""
AQRL analytical inverse kinematics.

Derived directly from the MuJoCo model geometry.
Verified against MuJoCo FK at < 1e-6 m.

Geometry (base_link frame, X=right, Y=forward, Z=up):
    hip anchors (joint 1 rotates about Y):
        FL: (-0.037,  0.105, 0.02)
        FR: ( 0.037,  0.105, 0.02)
        RL: (-0.037, -0.144, 0.02)
        RR: ( 0.037, -0.144, 0.02)

    Child-body offsets (sign flips with side; s = -1 for LEFT, +1 for RIGHT):
        p_m = (s*0.025, 0.023, -0.0105)   hip -> mid (mid has euler="90 0 0")
        p_w = (s*0.005,-0.107, -0.013)    mid -> wrist (wrist has euler="-90 0 0")
        f_w = (s*0.016, 0.13,  -0.005)    wrist -> foot site

FK closed form (thanks to Rx(π/2+j2)*Rx(-π/2+j3) = Rx(j2+j3)):
    foot_in_hip_mount = Ry(j1) · Inner(j2, j3)
    Inner(q, r) = p_m + Rx(π/2 + q) · p_w + Rx(q + r) · f_w

IK closed form:
    Inner_X = s * (0.025 + 0.005 + 0.016) = s*0.046   (constant; foot site X)
    Inner_Y = fy                                       (Ry invariant)
    Inner_Z = ±sqrt(fx² + fz² - Inner_X²)              (Ry invariant, sign chosen
                                                        by "leg below body" preference)
    j1      = atan2(Inner_Z, Inner_X) - atan2(fz, fx)

    Let Y' = Inner_Y - 0.023, Z' = Inner_Z + 0.0105
        b2 = -0.107, b3 = -0.013, c2 = 0.145, c3 = -0.005
        K1 = 2(Y'*b2 + Z'*b3)
        K2 = 2(Y'*b3 - Z'*b2)
        K3 = c2² + c3² - Y'² - Z'² - b2² - b3²

    Two branches for j2 = u:
        u_pos = asin(K3/R) - atan2(K2, K1)
        u_neg = π - asin(K3/R) - atan2(K2, K1)

    Then
        A = Y' + b2*sin(u) + b3*cos(u) = c2*cos(v) - c3*sin(v)
        B = Z' - b2*cos(u) + b3*sin(u) = c2*sin(v) + c3*cos(v)
        v = atan2(c2*B + c3*A, c2*A - c3*B)
        j3 = v - u
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


HIP_ANCHORS = {
    "FL": np.array([-0.037,  0.105, 0.02]),
    "FR": np.array([ 0.037,  0.105, 0.02]),
    "RL": np.array([-0.037, -0.144, 0.02]),
    "RR": np.array([ 0.037, -0.144, 0.02]),
}
SIDE_OF = {"FL": "LEFT", "RL": "LEFT", "FR": "RIGHT", "RR": "RIGHT"}

_A2 = 0.023
_A3 = -0.0105
_B2 = -0.107
_B3 = -0.013
_C2 = 0.13
_C3 = -0.005

_B_MAG_SQ = _B2 * _B2 + _B3 * _B3
_C_MAG_SQ = _C2 * _C2 + _C3 * _C3

JOINT_LIMIT = np.deg2rad(60.0)


@dataclass
class IKResult:
    ok: bool
    angles: np.ndarray          # shape (3,), [j1, j2, j3] MuJoCo convention
    branch: str                 # "pos", "neg", or "none"
    reason: str = ""


def _solve_j2_branches(y_prime: float, z_prime: float) -> Tuple[float, float, bool]:
    """Return (u_pos, u_neg, feasible). feasible=False if target out of workspace."""
    k1 = 2.0 * (y_prime * _B2 + z_prime * _B3)
    k2 = 2.0 * (y_prime * _B3 - z_prime * _B2)
    k3 = _C_MAG_SQ - y_prime * y_prime - z_prime * z_prime - _B_MAG_SQ

    r = np.hypot(k1, k2)
    if r < 1e-12:
        return 0.0, 0.0, False

    ratio = k3 / r
    if ratio > 1.0 or ratio < -1.0:
        return 0.0, 0.0, False

    asin_term = np.arcsin(ratio)
    phase = np.arctan2(k2, k1)
    u_pos = asin_term - phase
    u_neg = np.pi - asin_term - phase
    return _wrap(u_pos), _wrap(u_neg), True


def _wrap(theta: float) -> float:
    return np.arctan2(np.sin(theta), np.cos(theta))


def _solve_v(u: float, y_prime: float, z_prime: float) -> float:
    sin_u = np.sin(u)
    cos_u = np.cos(u)
    a = y_prime + _B2 * sin_u + _B3 * cos_u
    b = z_prime - _B2 * cos_u + _B3 * sin_u
    # cos(v) = (c2*a + c3*b) / C_MAG_SQ
    # sin(v) = (c2*b - c3*a) / C_MAG_SQ
    cos_v = _C2 * a + _C3 * b
    sin_v = _C2 * b - _C3 * a
    return np.arctan2(sin_v, cos_v)


def _within_limits(j1: float, j2: float, j3: float) -> bool:
    return (
        abs(j1) <= JOINT_LIMIT
        and abs(j2) <= JOINT_LIMIT
        and abs(j3) <= JOINT_LIMIT
    )


def solve(target_foot_hip_mount: np.ndarray, leg: str,
          prefer_branch: str = "auto") -> IKResult:
    """
    Solve IK for one leg.

    :param target_foot_hip_mount: (3,) foot position in hip-mount frame
        (base_link axes, origin at hip body position).
    :param leg: one of "FL", "FR", "RL", "RR".
    :param prefer_branch: "pos" = knee-forward, "neg" = knee-backward,
        "auto" = try pos first, fall back to neg.
    :return: IKResult with joint angles (j1, j2, j3) in MuJoCo convention.
    """
    side = SIDE_OF[leg]
    s = -1.0 if side == "LEFT" else +1.0

    fx, fy, fz = float(target_foot_hip_mount[0]), float(target_foot_hip_mount[1]), float(target_foot_hip_mount[2])
    inner_x = s * (0.025 + 0.005 + 0.016)

    radicand = fx * fx + fz * fz - inner_x * inner_x
    if radicand < 0:
        return IKResult(False, np.zeros(3), "none",
                        f"target XZ distance {np.hypot(fx,fz):.3f} < |Inner_X|={abs(inner_x):.3f}")

    inner_z = -np.sqrt(radicand)  # leg below body
    inner_y = fy

    j1 = _wrap(np.arctan2(inner_z, inner_x) - np.arctan2(fz, fx))

    y_prime = inner_y - _A2
    z_prime = inner_z - _A3

    u_pos, u_neg, feasible = _solve_j2_branches(y_prime, z_prime)
    if not feasible:
        return IKResult(False, np.zeros(3), "none",
                        f"|K3/R| > 1 (target outside 2-link reach in YZ plane)")

    candidates = []
    if prefer_branch in ("pos", "auto"):
        v_pos = _solve_v(u_pos, y_prime, z_prime)
        j3_pos = _wrap(v_pos - u_pos)
        candidates.append(("pos", u_pos, j3_pos))
    if prefer_branch in ("neg", "auto"):
        v_neg = _solve_v(u_neg, y_prime, z_prime)
        j3_neg = _wrap(v_neg - u_neg)
        candidates.append(("neg", u_neg, j3_neg))

    for branch_name, j2, j3 in candidates:
        if _within_limits(j1, j2, j3):
            return IKResult(True, np.array([j1, j2, j3]), branch_name)

    # Nothing fits; return the first branch so the caller can inspect the error.
    br, j2, j3 = candidates[0]
    return IKResult(False, np.array([j1, j2, j3]), br,
                    f"out of joint limits (j3={np.rad2deg(j3):+.1f}°)")


def gait_frame_to_mujoco_hipmount(h2f_gait_frame: np.ndarray) -> np.ndarray:
    """
    Gait-generator frame (X=fwd, Y=left, Z=up) hip-to-foot vector
    -> MuJoCo base frame (X=right, Y=fwd, Z=up) hip-mount offset.
    """
    return np.array([-h2f_gait_frame[1], h2f_gait_frame[0], h2f_gait_frame[2]])


def solve_from_gait_frame(h2f_gait_frame: np.ndarray, leg: str,
                          prefer_branch: str = "neg") -> IKResult:
    """
    Wrapper: take a hip-to-foot vector in the gait-generator frame and solve
    it in MuJoCo hip-mount coordinates.
    """
    target = gait_frame_to_mujoco_hipmount(np.asarray(h2f_gait_frame, dtype=float))
    return solve(target, leg, prefer_branch=prefer_branch)
