import math
import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from src.controllers.trot_planner import TrotPlanner, mujoco_quat_to_rpy
from src.robot.default_pose import DEFAULT_STANDING_QPOS_JOINTS
from src.sensors.imu_observation import normalize_imu_observation


class TrotImuEnv(gym.Env):
    """Residual locomotion env with robot-realistic IMU-style observation.

    Observation is intentionally limited to signals we can reproduce on hardware:
      - roll
      - pitch
      - yaw rate
      - integrated yaw error
      - gait phase
      - previous action
      - command (target forward velocity, target yaw rate)

    Policy learns small residuals on top of the analytical open-loop trot that
    currently works best on the physical robot.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        randomize_reset=False,
        random_lateral_range=0.0,
        random_yaw_range_deg=0.0,
        control_dt=0.04,
        base_step_length=0.030,
        base_step_velocity=0.30,
        base_clearance=0.040,
        base_penetration=0.003,
        target_forward_velocity=0.10,
        target_yaw_rate=0.0,
        random_forward_velocity_range=0.0,
        random_forward_velocity_min=0.0,
        zero_forward_probability=0.0,
        random_yaw_command_range=0.0,
        random_yaw_command_min_abs=0.0,
        straight_command_probability=0.0,
        forward_velocity_reward_weight=1.5,
        forward_velocity_error_scale=0.08,
        yaw_rate_error_weight=3.0,
        yaw_error_weight=2.5,
        integrated_yaw_error_weight=1.5,
        lateral_error_weight=0.8,
        lateral_velocity_weight=0.8,
        height_target=0.19,
        height_error_weight=2.0,
        yaw_residual_scale=0.020,
        action_weight=0.08,
        gait_action_weight=0.10,
        yaw_action_weight=0.25,
        lateral_body_action_weight=1.0,
        lift_action_weight=0.15,
        action_smoothness_weight=0.18,
        action_saturation_weight=3.0,
        max_lateral_error=0.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.randomize_reset = bool(randomize_reset)
        self.random_lateral_range = float(random_lateral_range)
        self.random_yaw_range = math.radians(float(random_yaw_range_deg))

        self.xml_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "assets",
                "mujoco",
                "xml",
                "aqrl_scene.xml",
            )
        )
        self.model = self._load_model()
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = float(control_dt)
        self.dt = self.control_dt
        self.frame_skip = max(1, int(round(self.control_dt / self.sim_dt)))

        # Baseline gait used by the physical open-loop trot script.
        self.base_step_length = float(base_step_length)
        self.base_step_velocity = float(base_step_velocity)
        self.base_clearance = float(base_clearance)
        self.base_penetration = float(base_penetration)

        self.fixed_target_forward_velocity = float(target_forward_velocity)
        self.fixed_target_yaw_rate = float(target_yaw_rate)
        self.target_forward_velocity = self.fixed_target_forward_velocity
        self.target_yaw_rate = self.fixed_target_yaw_rate

        self.random_forward_velocity_range = float(random_forward_velocity_range)
        self.random_forward_velocity_min = float(random_forward_velocity_min)
        self.zero_forward_probability = float(zero_forward_probability)
        self.random_yaw_command_range = float(random_yaw_command_range)
        self.random_yaw_command_min_abs = float(random_yaw_command_min_abs)
        self.straight_command_probability = float(straight_command_probability)

        self.forward_velocity_reward_weight = float(forward_velocity_reward_weight)
        self.forward_velocity_error_scale = float(forward_velocity_error_scale)
        self.yaw_rate_error_weight = float(yaw_rate_error_weight)
        self.yaw_error_weight = float(yaw_error_weight)
        self.integrated_yaw_error_weight = float(integrated_yaw_error_weight)
        self.lateral_error_weight = float(lateral_error_weight)
        self.lateral_velocity_weight = float(lateral_velocity_weight)
        self.height_target = float(height_target)
        self.height_error_weight = float(height_error_weight)
        self.yaw_residual_scale = float(yaw_residual_scale)
        self.action_weight = float(action_weight)
        self.gait_action_weight = float(gait_action_weight)
        self.yaw_action_weight = float(yaw_action_weight)
        self.lateral_body_action_weight = float(lateral_body_action_weight)
        self.lift_action_weight = float(lift_action_weight)
        self.action_smoothness_weight = float(action_smoothness_weight)
        self.action_saturation_weight = float(action_saturation_weight)
        self.max_lateral_error = float(max_lateral_error)

        # Action = residual controller knobs
        # [step_length, step_velocity, clearance, lateral_fraction, yaw_residual,
        #  body_y_offset, FL_lift_scale, FR_lift_scale, RL_lift_scale, RR_lift_scale]
        self.action_dim = 10
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # Observation:
        # [normalized roll, pitch, yaw_rate, integrated_yaw_error, phase_sin, phase_cos,
        #  prev_action(10), command(2)]
        self.obs_dim = 4 + 2 + self.action_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.planner = TrotPlanner(dt=self.control_dt, Tswing=0.2)
        self.step_count = 0
        self.max_episode_steps = 1000
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.yaw_ref = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.integrated_yaw_error = 0.0

    def _load_model(self):
        xml_dir = os.path.dirname(self.xml_path)
        old_cwd = os.getcwd()
        os.chdir(xml_dir)
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        os.chdir(old_cwd)
        return model

    def _set_base_yaw(self, yaw):
        half = 0.5 * yaw
        self.data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]

    def _apply_reset_randomization(self):
        if not self.randomize_reset:
            return
        if self.random_lateral_range > 0.0:
            self.data.qpos[0] += self.np_random.uniform(
                -self.random_lateral_range,
                self.random_lateral_range,
            )
        if self.random_yaw_range > 0.0:
            yaw = mujoco_quat_to_rpy(self.data.qpos[3:7])[2]
            yaw += self.np_random.uniform(-self.random_yaw_range, self.random_yaw_range)
            self._set_base_yaw(yaw)
        mujoco.mj_forward(self.model, self.data)

    def _sample_commands(self):
        self.target_forward_velocity = self.fixed_target_forward_velocity
        self.target_yaw_rate = self.fixed_target_yaw_rate

        if self.random_forward_velocity_range > 0.0:
            if (
                self.random_forward_velocity_min > 0.0
                or self.zero_forward_probability > 0.0
            ):
                if self.np_random.random() < np.clip(self.zero_forward_probability, 0.0, 1.0):
                    self.target_forward_velocity = 0.0
                else:
                    min_forward = min(
                        max(0.0, self.random_forward_velocity_min),
                        self.random_forward_velocity_range,
                    )
                    self.target_forward_velocity = self.np_random.uniform(
                        min_forward, self.random_forward_velocity_range
                    )
            else:
                self.target_forward_velocity = self.np_random.uniform(
                    0.0, self.random_forward_velocity_range
                )

        if self.random_yaw_command_range > 0.0:
            if (
                self.random_yaw_command_min_abs > 0.0
                or self.straight_command_probability > 0.0
            ):
                if self.np_random.random() < np.clip(self.straight_command_probability, 0.0, 1.0):
                    self.target_yaw_rate = 0.0
                else:
                    min_abs = min(
                        max(0.0, self.random_yaw_command_min_abs),
                        self.random_yaw_command_range,
                    )
                    magnitude = self.np_random.uniform(min_abs, self.random_yaw_command_range)
                    sign = -1.0 if self.np_random.random() < 0.5 else 1.0
                    self.target_yaw_rate = sign * magnitude
            else:
                self.target_yaw_rate = self.np_random.uniform(
                    -self.random_yaw_command_range,
                    self.random_yaw_command_range,
                )

    def _desired_yaw(self):
        return self.yaw_ref + self.target_yaw_rate * self.step_count * self.dt

    def _yaw_error(self, yaw):
        desired_yaw = self._desired_yaw()
        return math.atan2(math.sin(yaw - desired_yaw), math.cos(yaw - desired_yaw))

    def _desired_xy(self):
        t = self.step_count * self.dt
        v = self.target_forward_velocity
        w = self.target_yaw_rate
        if abs(w) < 1e-6:
            local_right = 0.0
            local_forward = v * t
        else:
            theta = w * t
            radius = v / w
            local_right = -radius * (1.0 - math.cos(theta))
            local_forward = radius * math.sin(theta)
        right_axis = np.array([math.cos(self.yaw_ref), math.sin(self.yaw_ref)])
        forward_axis = np.array([-math.sin(self.yaw_ref), math.cos(self.yaw_ref)])
        start_xy = np.array([self.start_x, self.start_y])
        return start_xy + right_axis * local_right + forward_axis * local_forward

    def _lateral_error(self):
        desired_xy = self._desired_xy()
        current_xy = np.array([float(self.data.qpos[0]), float(self.data.qpos[1])])
        right_axis = np.array([math.cos(self.yaw_ref), math.sin(self.yaw_ref)])
        return float(np.dot(current_xy - desired_xy, right_axis))

    def _phase_features(self):
        phase = 2.0 * math.pi * self.step_count * self.dt / 0.4
        return np.array([math.sin(phase), math.cos(phase)], dtype=np.float32)

    def _apply_leg_lift_scales(self, qpos_targets, stand_qpos, lift_scales):
        scaled = np.asarray(qpos_targets, dtype=np.float32).copy()
        if np.allclose(lift_scales, 1.0):
            return scaled

        knee_indices = [2, 8, 5, 11]
        for joint_idx, scale in zip(knee_indices, lift_scales):
            scaled[joint_idx] = stand_qpos[joint_idx] + (
                scaled[joint_idx] - stand_qpos[joint_idx]
            ) * float(scale)
        return scaled

    def _map_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        return {
            "step_length": float(np.clip(self.base_step_length + 0.008 * action[0], 0.022, 0.040)),
            "step_velocity": float(np.clip(self.base_step_velocity + 0.04 * action[1], 0.24, 0.36)),
            "clearance": float(np.clip(self.base_clearance + 0.008 * action[2], 0.030, 0.052)),
            "lateral_fraction": float(0.006 * action[3]),
            "yaw_residual": float(self.yaw_residual_scale * action[4]),
            "body_y_offset": float(0.003 * action[5]),
            "lift_scales": np.clip(
                1.0 + 0.12 * np.asarray(action[6:10], dtype=np.float32),
                0.88,
                1.15,
            ).astype(np.float32),
        }

    def _get_obs(self):
        roll, pitch, yaw = mujoco_quat_to_rpy(self.data.qpos[3:7])
        yaw_rate = float(self.data.qvel[5])
        imu_obs = normalize_imu_observation(
            roll,
            pitch,
            yaw_rate,
            self.integrated_yaw_error,
        )
        command = np.array(
            [self.target_forward_velocity, self.target_yaw_rate],
            dtype=np.float32,
        )
        obs = np.concatenate(
            [
                imu_obs,
                self._phase_features(),
                self.prev_action.astype(np.float32),
                command,
            ]
        ).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del options

        self.step_count = 0
        self.prev_action[:] = 0.0
        self.last_action[:] = 0.0
        self.integrated_yaw_error = 0.0
        self._sample_commands()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:19] = DEFAULT_STANDING_QPOS_JOINTS
        self.data.ctrl[:] = DEFAULT_STANDING_QPOS_JOINTS
        mujoco.mj_forward(self.model, self.data)

        for _ in range(int(1.0 / self.sim_dt)):
            self.data.ctrl[:] = DEFAULT_STANDING_QPOS_JOINTS
            mujoco.mj_step(self.model, self.data)

        self._apply_reset_randomization()
        self.yaw_ref = mujoco_quat_to_rpy(self.data.qpos[3:7])[2]
        self.start_x = float(self.data.qpos[0])
        self.start_y = float(self.data.qpos[1])
        self.planner.reset(initial_targets=DEFAULT_STANDING_QPOS_JOINTS)
        return self._get_obs(), {}

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32)
        raw_action = np.clip(raw_action, -1.0, 1.0)
        smoothed_action = 0.8 * self.prev_action + 0.2 * raw_action
        params = self._map_action(smoothed_action)

        yaw_rate = self.target_yaw_rate + params["yaw_residual"]
        qpos_targets = self.planner.compute_joint_targets(
            step_length=params["step_length"],
            step_velocity=params["step_velocity"],
            clearance_height=params["clearance"],
            penetration_depth=self.base_penetration,
            yaw_rate=yaw_rate,
            lateral_fraction=params["lateral_fraction"],
            body_pos_offset=[0.0, params["body_y_offset"], 0.0],
        )
        qpos_targets = self._apply_leg_lift_scales(
            qpos_targets,
            DEFAULT_STANDING_QPOS_JOINTS,
            params["lift_scales"],
        )

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = qpos_targets
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.integrated_yaw_error += float(self.data.qvel[5]) * self.dt
        self.integrated_yaw_error = float(np.clip(self.integrated_yaw_error, -0.8, 0.8))

        obs = self._get_obs()
        reward = self._reward(smoothed_action, raw_action)
        terminated = self._terminated()
        truncated = self.step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        self.last_action = raw_action.copy()
        self.prev_action = smoothed_action.copy()
        info = {
            **params,
            "commanded_yaw_rate": yaw_rate,
            "target_forward_velocity": self.target_forward_velocity,
            "target_yaw_rate": self.target_yaw_rate,
            "integrated_yaw_error": self.integrated_yaw_error,
            "yaw_error": self._yaw_error(mujoco_quat_to_rpy(self.data.qpos[3:7])[2]),
            "lateral_error": self._lateral_error(),
            "joint_targets": qpos_targets,
        }
        return obs, reward, terminated, truncated, info

    def _reward(self, action, raw_action):
        del raw_action
        roll, pitch, yaw = mujoco_quat_to_rpy(self.data.qpos[3:7])
        height = float(self.data.qpos[2])
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        yaw_rate = float(self.data.qvel[5])
        lateral_err = self._lateral_error()
        yaw_err = self._yaw_error(yaw)

        forward_velocity_error = vy - self.target_forward_velocity
        velocity_error = forward_velocity_error / max(1e-6, self.forward_velocity_error_scale)
        velocity_reward = self.forward_velocity_reward_weight * math.exp(
            -(velocity_error * velocity_error)
        )
        alive = 0.05

        penalty = 0.0
        penalty += self.yaw_rate_error_weight * abs(yaw_rate - self.target_yaw_rate)
        penalty += self.yaw_error_weight * abs(yaw_err)
        penalty += self.lateral_error_weight * abs(lateral_err)
        penalty += 1.2 * abs(roll) + 0.8 * abs(pitch)
        penalty += self.height_error_weight * abs(height - self.height_target)
        penalty += self.action_weight * float(np.sum(np.square(action)))
        penalty += self.gait_action_weight * float(np.sum(np.square(action[0:3])))
        penalty += self.yaw_action_weight * float(action[4] * action[4])
        penalty += self.lateral_body_action_weight * float(action[3] * action[3] + action[5] * action[5])
        penalty += self.lift_action_weight * float(np.sum(np.square(action[6:10])))
        action_saturation = np.maximum(0.0, np.abs(action) - 0.85)
        penalty += self.action_saturation_weight * float(np.sum(np.square(action_saturation)))
        penalty += self.action_smoothness_weight * float(np.sum(np.square(action - self.last_action)))
        penalty += self.integrated_yaw_error_weight * abs(self.integrated_yaw_error)
        penalty += self.lateral_velocity_weight * abs(vx)

        return float(velocity_reward + alive - penalty)

    def _terminated(self):
        roll, pitch, yaw = mujoco_quat_to_rpy(self.data.qpos[3:7])
        yaw_err = self._yaw_error(yaw)
        if float(self.data.qpos[2]) < 0.11:
            return True
        if abs(roll) > np.deg2rad(45.0) or abs(pitch) > np.deg2rad(45.0):
            return True
        if abs(yaw_err) > np.deg2rad(90.0):
            return True
        if self.max_lateral_error > 0.0 and abs(self._lateral_error()) > self.max_lateral_error:
            return True
        return False

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
