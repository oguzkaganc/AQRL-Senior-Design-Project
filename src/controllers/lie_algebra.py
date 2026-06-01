"""Homogeneous transform utilities used by the AQRL gait planner."""

import numpy as np


def RpToTrans(R, p):
    """Rotation matrix + position vector -> 4x4 homogeneous transform."""
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]


def TransToRp(T):
    """4x4 homogeneous transform -> (3x3 rotation, 3-vector position)."""
    T = np.array(T)
    return T[0:3, 0:3], T[0:3, 3]


def TransInv(T):
    """Efficient inverse of a homogeneous transform (uses R^T, not np.linalg.inv)."""
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]


def RPY(roll, pitch, yaw):
    """Roll-Pitch-Yaw -> 4x4 homogeneous rotation (translation = 0)."""
    Roll = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll),  np.cos(roll), 0],
        [0, 0, 0, 1],
    ])
    Pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1],
    ])
    Yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw),  np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return np.matmul(np.matmul(Roll, Pitch), Yaw)


def RotateTranslate(rotation, position):
    """Creates a transform: first rotate, then translate."""
    trans = np.eye(4)
    trans[0, 3] = position[0]
    trans[1, 3] = position[1]
    trans[2, 3] = position[2]
    return np.dot(rotation, trans)


def TransformVector(xyz_coord, rotation, translation):
    """Transform a 3D vector by rotation then translation."""
    xyz_vec = np.append(xyz_coord, 1.0)
    Transformed = np.dot(RotateTranslate(rotation, translation), xyz_vec)
    return Transformed[:3]
