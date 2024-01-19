import numpy as np
import load_data as ld
import matplotlib.pyplot as plt

fileNumber = '22'

imuFileName = 'data/imu' + fileNumber
encoderFileName = 'data/Encoders' + fileNumber
lidarFileName = 'data/Hokuyo' + fileNumber

# Load Lidar, IMU and encoder data
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_ts = ld.get_imu(imuFileName)
FR, FL, RR, RL, enc_ts = ld.get_encoder(encoderFileName)
lidar = ld.get_lidar(lidarFileName)


# Constants
wheelRadius = 0.127  # meters
wheelBase =  69 # cm
ticksPerRevolution = 360

# Calculate the distance traveled per tick
distancePerTick = 2 * np.pi * wheelRadius / ticksPerRevolution * 100  # in centimeters

# Calculate wheel odometry
R = (FR + RR) / 2 * distancePerTick
L = (FL + RL) / 2 * distancePerTick
F = (R + L) / 2
ths = (R - L) / wheelBase
thSum = np.zeros(len(ths))
for i in range(len(ths)):
    thSum[i] = np.sum(ths[:i])

# Initialize position arrays
xs = [0]
ys = [0]

# Iterate through the encoder data
for t in range(len(R)):
    th = thSum[t]
    dx = np.cos(th) * F[t]
    dy = np.sin(th) * F[t]
    x = xs[-1] + dx
    y = ys[-1] + dy
    xs.append(x)
    ys.append(y)

    # Get the LIDAR data at the current time step
    if t < len(lidar):
        sc = lidar[t]["scan"].flatten() * 100  # Convert LIDAR data to centimeters
        ang = lidar[t]["angle"].flatten()

        # Plot LIDAR data every 100 steps
        if t % 100 == 0:
            plt.scatter(np.cos(ang + th) * sc + x, np.sin(ang + th) * sc + y, c="gray", s=1)

# Plot the robot's path
plt.plot(xs, ys, c='red', linewidth=2)
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.title("Robot's Path and LIDAR Data (map" + fileNumber + ')')
plt.axis("equal")
plt.show()