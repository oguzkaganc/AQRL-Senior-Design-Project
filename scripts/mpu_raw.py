import time

from smbus2 import SMBus


MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

ACCEL_SCALE = 16384.0  # +/-2g
GYRO_SCALE = 131.0     # +/-250 deg/s


def read_word_2c(bus, reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    value = (high << 8) | low
    if value >= 0x8000:
        value = -((65535 - value) + 1)
    return value


def main():
    with SMBus(1) as bus:
        # Wake up MPU6050
        bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)

        print("MPU6050 raw stream started. Press Ctrl+C to stop.")
        while True:
            ax = read_word_2c(bus, ACCEL_XOUT_H + 0) / ACCEL_SCALE
            ay = read_word_2c(bus, ACCEL_XOUT_H + 2) / ACCEL_SCALE
            az = read_word_2c(bus, ACCEL_XOUT_H + 4) / ACCEL_SCALE

            gx = read_word_2c(bus, GYRO_XOUT_H + 0) / GYRO_SCALE
            gy = read_word_2c(bus, GYRO_XOUT_H + 2) / GYRO_SCALE
            gz = read_word_2c(bus, GYRO_XOUT_H + 4) / GYRO_SCALE

            print(
                f"acc[g]=({ax:+.3f}, {ay:+.3f}, {az:+.3f})   "
                f"gyro[dps]=({gx:+.2f}, {gy:+.2f}, {gz:+.2f})"
            )
            time.sleep(0.05)


if __name__ == "__main__":
    main()
