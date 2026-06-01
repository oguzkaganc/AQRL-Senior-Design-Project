import math
import time

from smbus2 import SMBus


MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

ACCEL_SCALE = 16384.0  # +/-2g
GYRO_SCALE = 131.0     # +/-250 deg/s

CALIBRATION_SECONDS = 3.0
PRINT_DT = 0.05
ALPHA = 0.98


def read_word_2c(bus, reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    value = (high << 8) | low
    if value >= 0x8000:
        value = -((65535 - value) + 1)
    return value


def read_accel_gyro(bus):
    ax = read_word_2c(bus, ACCEL_XOUT_H + 0) / ACCEL_SCALE
    ay = read_word_2c(bus, ACCEL_XOUT_H + 2) / ACCEL_SCALE
    az = read_word_2c(bus, ACCEL_XOUT_H + 4) / ACCEL_SCALE

    gx = read_word_2c(bus, GYRO_XOUT_H + 0) / GYRO_SCALE
    gy = read_word_2c(bus, GYRO_XOUT_H + 2) / GYRO_SCALE
    gz = read_word_2c(bus, GYRO_XOUT_H + 4) / GYRO_SCALE
    return ax, ay, az, gx, gy, gz


def accel_to_pitch_roll_deg(ax, ay, az):
    # Based on the measured mounting:
    #   Y axis is vertical in neutral pose
    #   X changes with forward/back tilt
    #   Z changes with left/right roll
    pitch_deg = math.degrees(math.atan2(ax, -ay))
    roll_deg = math.degrees(math.atan2(az, -ay))
    return pitch_deg, roll_deg


def calibrate_gyro_bias(bus, duration_s=CALIBRATION_SECONDS):
    print(f"Starting gyro bias calibration ({duration_s:.1f}s). Keep the sensor still.")
    samples = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        _, _, _, gx, gy, gz = read_accel_gyro(bus)
        samples.append((gx, gy, gz))
        time.sleep(0.005)

    n = max(1, len(samples))
    bx = sum(s[0] for s in samples) / n
    by = sum(s[1] for s in samples) / n
    bz = sum(s[2] for s in samples) / n
    print(
        f"Gyro bias[dps]=({bx:+.3f}, {by:+.3f}, {bz:+.3f})  "
        f"samples={n}"
    )
    return bx, by, bz


def main():
    with SMBus(1) as bus:
        bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)

        bias_gx, bias_gy, bias_gz = calibrate_gyro_bias(bus)

        ax, ay, az, gx, gy, gz = read_accel_gyro(bus)
        pitch_deg, roll_deg = accel_to_pitch_roll_deg(ax, ay, az)
        yaw_rate_dps = gy - bias_gy

        print("MPU6050 filtered stream started. Press Ctrl+C to stop.")

        prev_t = time.time()
        while True:
            ax, ay, az, gx, gy, gz = read_accel_gyro(bus)
            now = time.time()
            dt = max(1e-4, now - prev_t)
            prev_t = now

            gx -= bias_gx
            gy -= bias_gy
            gz -= bias_gz

            accel_pitch_deg, accel_roll_deg = accel_to_pitch_roll_deg(ax, ay, az)

            # Based on the mounting discovered from the log:
            #   gyro_z mainly tracks pitch motion
            #   gyro_x mainly tracks roll motion
            pitch_deg = ALPHA * (pitch_deg + gz * dt) + (1.0 - ALPHA) * accel_pitch_deg
            roll_deg = ALPHA * (roll_deg + gx * dt) + (1.0 - ALPHA) * accel_roll_deg
            yaw_rate_dps = gy

            print(
                f"pitch={pitch_deg:+7.2f} deg   "
                f"roll={roll_deg:+7.2f} deg   "
                f"yaw_rate={yaw_rate_dps:+7.2f} dps   "
                f"acc=({ax:+.3f}, {ay:+.3f}, {az:+.3f})"
            )
            time.sleep(PRINT_DT)


if __name__ == "__main__":
    main()
