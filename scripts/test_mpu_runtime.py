import os
import sys
import time


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.sensors.mpu6050 import MPU6050


def main():
    imu = MPU6050()
    imu.open()

    print("MPU6050 runtime test: calibrating...")
    bias = imu.calibrate()
    print("bias:", {k: round(v, 3) for k, v in bias.items()})
    print("MPU6050 runtime stream started. Press Ctrl+C to stop.")

    while True:
        sample = imu.read()
        print(
            f"real_pitch={sample['pitch_rad']:+7.3f} rad   "
            f"real_roll={sample['roll_rad']:+7.3f} rad   "
            f"real_yaw_rate={sample['yaw_rate_rad_s']:+7.3f} rad/s   "
            f"mujoco=({sample['mujoco_roll_rad']:+.3f}, "
            f"{sample['mujoco_pitch_rad']:+.3f}, "
            f"{sample['mujoco_yaw_rate_rad_s']:+.3f})   "
            f"obs={sample['policy_imu_obs'].round(3).tolist()}"
        )
        time.sleep(0.05)


if __name__ == "__main__":
    main()
