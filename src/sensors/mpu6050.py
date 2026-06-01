import math
import time

from smbus2 import SMBus

from src.sensors.imu_observation import dps_to_rad_s, deg_to_rad, normalize_imu_observation


class MPU6050:
    MPU_ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    ACCEL_SCALE = 16384.0  # +/-2g
    GYRO_SCALE = 131.0     # +/-250 deg/s

    def __init__(
        self,
        bus_id=1,
        alpha=0.98,
        calibration_seconds=3.0,
    ):
        self.bus_id = int(bus_id)
        self.alpha = float(alpha)
        self.calibration_seconds = float(calibration_seconds)

        self.bus = None
        self.bias_gx = 0.0
        self.bias_gy = 0.0
        self.bias_gz = 0.0

        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.yaw_rate_dps = 0.0
        self.integrated_yaw_error_rad = 0.0
        self.prev_t = None

    def _read_word_2c(self, reg):
        high = self.bus.read_byte_data(self.MPU_ADDR, reg)
        low = self.bus.read_byte_data(self.MPU_ADDR, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value

    def _read_accel_gyro(self):
        ax = self._read_word_2c(self.ACCEL_XOUT_H + 0) / self.ACCEL_SCALE
        ay = self._read_word_2c(self.ACCEL_XOUT_H + 2) / self.ACCEL_SCALE
        az = self._read_word_2c(self.ACCEL_XOUT_H + 4) / self.ACCEL_SCALE

        gx = self._read_word_2c(self.GYRO_XOUT_H + 0) / self.GYRO_SCALE
        gy = self._read_word_2c(self.GYRO_XOUT_H + 2) / self.GYRO_SCALE
        gz = self._read_word_2c(self.GYRO_XOUT_H + 4) / self.GYRO_SCALE
        return ax, ay, az, gx, gy, gz

    @staticmethod
    def _accel_to_pitch_roll_deg(ax, ay, az):
        # Calibrated from real mounting tests:
        # Y axis is roughly vertical in neutral pose.
        # X changes with pitch, Z changes with roll.
        pitch_deg = math.degrees(math.atan2(ax, -ay))
        roll_deg = math.degrees(math.atan2(az, -ay))
        return pitch_deg, roll_deg

    def open(self):
        if self.bus is not None:
            return
        self.bus = SMBus(self.bus_id)
        self.bus.write_byte_data(self.MPU_ADDR, self.PWR_MGMT_1, 0)
        time.sleep(0.1)

    def close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def calibrate(self):
        if self.bus is None:
            self.open()

        samples = []
        t0 = time.time()
        while time.time() - t0 < self.calibration_seconds:
            _, _, _, gx, gy, gz = self._read_accel_gyro()
            samples.append((gx, gy, gz))
            time.sleep(0.005)

        n = max(1, len(samples))
        self.bias_gx = sum(s[0] for s in samples) / n
        self.bias_gy = sum(s[1] for s in samples) / n
        self.bias_gz = sum(s[2] for s in samples) / n

        ax, ay, az, _, _, _ = self._read_accel_gyro()
        self.pitch_deg, self.roll_deg = self._accel_to_pitch_roll_deg(ax, ay, az)
        self.yaw_rate_dps = 0.0
        self.integrated_yaw_error_rad = 0.0
        self.prev_t = time.time()

        return {
            "bias_gx_dps": self.bias_gx,
            "bias_gy_dps": self.bias_gy,
            "bias_gz_dps": self.bias_gz,
        }

    def read(self):
        if self.bus is None:
            self.open()
        if self.prev_t is None:
            self.calibrate()

        ax, ay, az, gx, gy, gz = self._read_accel_gyro()
        now = time.time()
        dt = max(1e-4, now - self.prev_t)
        self.prev_t = now

        gx -= self.bias_gx
        gy -= self.bias_gy
        gz -= self.bias_gz

        accel_pitch_deg, accel_roll_deg = self._accel_to_pitch_roll_deg(ax, ay, az)

        # Calibrated from real tests:
        # gyro_z tracks pitch changes best, gyro_x tracks roll, gyro_y is yaw rate.
        self.pitch_deg = self.alpha * (self.pitch_deg + gz * dt) + (1.0 - self.alpha) * accel_pitch_deg
        self.roll_deg = self.alpha * (self.roll_deg + gx * dt) + (1.0 - self.alpha) * accel_roll_deg
        self.yaw_rate_dps = gy
        pitch_rad = deg_to_rad(self.pitch_deg)
        roll_rad = deg_to_rad(self.roll_deg)
        yaw_rate_rad_s = dps_to_rad_s(self.yaw_rate_dps)

        # Transform real mounting semantics into MuJoCo env convention:
        # - real pitch positive = nose/front down; MuJoCo roll_x positive = nose up
        # - real roll positive = left tilt; MuJoCo pitch_y positive = right tilt
        # - real yaw_rate positive = right turn; MuJoCo qvel[5] positive = left turn
        mujoco_roll_rad = -pitch_rad
        mujoco_pitch_rad = -roll_rad
        mujoco_yaw_rate_rad_s = -yaw_rate_rad_s

        self.integrated_yaw_error_rad += mujoco_yaw_rate_rad_s * dt
        self.integrated_yaw_error_rad = max(
            -0.8,
            min(0.8, self.integrated_yaw_error_rad),
        )
        policy_imu_obs = normalize_imu_observation(
            mujoco_roll_rad,
            mujoco_pitch_rad,
            mujoco_yaw_rate_rad_s,
            self.integrated_yaw_error_rad,
        )

        return {
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "yaw_rate_dps": self.yaw_rate_dps,
            "pitch_rad": pitch_rad,
            "roll_rad": roll_rad,
            "yaw_rate_rad_s": yaw_rate_rad_s,
            "mujoco_roll_rad": mujoco_roll_rad,
            "mujoco_pitch_rad": mujoco_pitch_rad,
            "mujoco_yaw_rate_rad_s": mujoco_yaw_rate_rad_s,
            "integrated_yaw_error_rad": self.integrated_yaw_error_rad,
            "policy_imu_obs": policy_imu_obs,
            "gyro_x_dps": gx,
            "gyro_y_dps": gy,
            "gyro_z_dps": gz,
            "acc_x_g": ax,
            "acc_y_g": ay,
            "acc_z_g": az,
            "dt": dt,
        }
