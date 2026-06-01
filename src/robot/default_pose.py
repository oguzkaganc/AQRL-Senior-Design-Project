"""
Default standing joint angles for AQRL. Computed by calling the live
TrotPlanner with zero motion, so the standing pose always matches the trot
pipeline's idea of "rest".
"""

from src.controllers.trot_planner import TrotPlanner


def _compute_standing_pose():
    planner = TrotPlanner(dt=0.002, Tswing=0.2)
    return planner.compute_joint_targets(
        step_length=0.0,
        step_velocity=0.0,
        clearance_height=0.0,
        penetration_depth=0.0,
    )


DEFAULT_STANDING_POSE = _compute_standing_pose()
DEFAULT_STANDING_QPOS_JOINTS = DEFAULT_STANDING_POSE.copy()
