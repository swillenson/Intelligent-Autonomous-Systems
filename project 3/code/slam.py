import numpy as np
import load_data as ld
import matplotlib.pyplot as plt
from MapUtils_fclad import getMapCellsFromRay_fclad

fileNumber = '21'

imuFileName = 'data/imu' + fileNumber
encoderFileName = 'data/Encoders' + fileNumber
lidarFileName = 'data/Hokuyo' + fileNumber

# Load Lidar, IMU and encoder data
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_ts = ld.get_imu(imuFileName)
FR, FL, RR, RL, enc_ts = ld.get_encoder(encoderFileName)
lidar = ld.get_lidar(lidarFileName)


def computeWeights(particles, lidarData, mapData):
    weights = np.zeros(len(particles))
    
    for i, particle in enumerate(particles):
        particleX, particleY, particleTheta = particle
        z = 0  # Initialize log odds ratio
        
        for angle, distance in zip(lidarData["angle"], lidarData["scan"]):
            # Convert LIDAR data to map coordinates
            mapX = particleX + distance * np.cos(angle + particleTheta)
            mapY = particleY + distance * np.sin(angle + particleTheta)
            
            # Check if the LIDAR hit is within the bounds of the map
            if mapX >= mapData["xmin"] and mapX < mapData["xmax"] and mapY >= mapData["ymin"] and mapY < mapData["ymax"]:
                # Calculate the grid index
                gridX = int((mapX - mapData["xmin"]) // mapData["res"])
                gridY = int((mapY - mapData["ymin"]) // mapData["res"])
                
                if mapData["map"][gridX, gridY] == 1:
                    z += 0.9  # Add 0.7 if the grid is occupied
                else:
                    z -= 0.1  # Subtract 0.3 if the grid is not occupied
                
                # Bound z between -10 and 10
                z = min(max(z, -10), 10)

        # Calculate Pwall for the particle and store it as the particle's weight
        Pwall = np.exp(z) / (1 + np.exp(z))
        weights[i] = Pwall
    
    # Normalize the weights so they sum to 1
    weights /= np.sum(weights)
    
    return weights

def resampleParticles(weights, minEffectiveParticles, particles):
    # Calculate nEffective
    weightsSum = np.sum(weights)
    weightsSquaredSum = np.sum(weights**2)
    nEffective = (weightsSum**2) / weightsSquaredSum

    # Check if resampling is needed
    if nEffective < minEffectiveParticles:
        return particles, weights

    # Perform resampling using the cumulative probability method
    cumWeights = np.cumsum(weights)
    newParticles = []
    for _ in range(len(particles)):
        randomNumber = np.random.rand()
        particleIndex = np.searchsorted(cumWeights, randomNumber)
        newParticle = particles[particleIndex].copy()  # Copy the selected particle
        newParticles.append(newParticle)

    newParticles = np.array(newParticles)
    newWeights = np.ones(len(newParticles)) / len(newParticles)  # Reset the weights

    return newParticles, newWeights


def createMap(mapResolution):
    # Define the dimensions of the map (in meters)
    mapWidth = 600  # meters
    mapHeight = 600  # meters

    # Calculate the number of grid cells in each dimension
    numCellsX = int(mapWidth / mapResolution)
    numCellsY = int(mapHeight / mapResolution)

    # Initialize the occupancy grid map with zeros (unknown)
    occupancyGridMap = np.zeros((numCellsX, numCellsY))

    # Create the mapData dictionary
    mapData = {
        "xmin": -mapWidth / 2,
        "ymin": -mapHeight / 2,
        "xmax": mapWidth / 2,
        "ymax": mapHeight / 2,
        "res": mapResolution,
        "map": occupancyGridMap
    }

    # Return the created mapData dictionary
    return mapData



def updateOccupancyGrid(particles, weights, lidarData, mapData):
    mapResolution = mapData['res']
    mapWidth = mapData['xmax'] - mapData['xmin']
    mapHeight = mapData['ymax'] - mapData['ymin']

    logOddsHit = 0.9
    logOddsMiss = -0.1
    maxMap = int((mapWidth + mapHeight) / mapResolution)

    for p, w in zip(particles, weights):
        x, y, theta = p

        # Transform LIDAR data to global coordinates
        lidarGlobal = lidarDataToGlobal(lidarData["scan"], lidarData["angle"], x, y, theta)

        xends = np.array([gx for gx, gy in lidarGlobal], dtype=np.int16)
        yends = np.array([gy for gx, gy in lidarGlobal], dtype=np.int16)

        # Convert the xends and yends arrays to 1-dimensional arrays of type short
        xends = xends.astype(np.int16).flatten()
        yends = yends.astype(np.int16).flatten()

        # Get map cells from ray using the Cython function
        ray_cells = getMapCellsFromRay_fclad(int(x), int(y), xends, yends, maxMap)

        for mx, my in ray_cells.T:
            # Update occupancy grid map based on LIDAR data
            if 0 <= mx < mapData["map"].shape[0] and 0 <= my < mapData["map"].shape[1]:
                mapData['map'][mx, my] += w * logOddsHit
            else:
                mapData['map'][mx, my] += w * logOddsMiss

    # Clip log-odds values to a specific range
    np.clip(mapData['map'], -10, 10, out=mapData['map'])

def lidarDataToGlobal(sc, ang, x, y, theta):
    lidarGlobal = []

    for r, angle in zip(sc, ang):
        gx = x + r * np.cos(theta + angle)
        gy = y + r * np.sin(theta + angle)
        lidarGlobal.append((gx, gy))

    return lidarGlobal

def globalToMapCoordinates(gx, gy, mapData):
    mx = int((gx - mapData['xmin']) / mapData['res'])
    my = int((gy - mapData['ymin']) / mapData['res'])

    return mx, my

def plotRobotAndParticles(xs, ys, particles):
    plt.scatter(*zip(*particles), c="blue", s=10, alpha=0.5, label='Particles')
    plt.plot(xs, ys, c='red', linewidth=2, label='Robot Path')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Robot's Path and Particles")
    plt.legend()
    plt.axis("equal")

def updateParticles(particles, F, thSum, t):
    # Get the current control inputs
    dx = np.cos(thSum[t]) * F[t]
    dy = np.sin(thSum[t]) * F[t]
    dtheta = thSum[t] - thSum[t - 1] if t > 0 else thSum[t]

    # Add Gaussian noise to the control input
    noiseSTD = [0.1, 0.1, 0.01]  
    noise_dx = np.random.normal(0, noiseSTD[0], len(particles))
    noise_dy = np.random.normal(0, noiseSTD[1], len(particles))
    noise_dth = np.random.normal(0, noiseSTD[2], len(particles))

    # Update the particles' states
    particles[:, 0] += dx + noise_dx
    particles[:, 1] += dy + noise_dy
    particles[:, 2] += dtheta + noise_dth

    return particles

def plot_map(mapData, step):
    plt.figure(figsize=(10, 10))
    plt.imshow(mapData['map'].T, origin='lower', cmap='gray_r', extent=[mapData['xmin'], mapData['xmax'], mapData['ymin'], mapData['ymax']])
    plt.title(f"Occupancy Grid Map at step {step}")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.show()

# Create the occupancy grid map
mapResolution = 0.1  # meters
mapData = createMap(mapResolution)


numParticles = 4
particles = np.zeros((numParticles, 3))  # x, y, theta
weights = np.ones(numParticles) / numParticles

minEffectiveParticles = 0.5 * numParticles

# Constants
wheelRadius = 0.127  # meters
wheelBase = 75  # cm
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

print("entering main loop")

count = 0
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
        
        # Update particles based on control_input (motion model)
        controlInput = np.array([dx, dy, ths[t]])
        particles = updateParticles(particles, F, thSum, t)
        # Update weights based on lidar_data and the occupancy grid map (measurement model)
        weights = computeWeights(particles, lidar[t], mapData)
        print(count)
        count += 1
        # Resample particles if necessary
        particles, weights = resampleParticles(weights, minEffectiveParticles, particles)

        # Update the occupancy grid map
        updateOccupancyGrid(particles, weights, lidar[t], mapData)

        # Plot the map every 100 steps
        # if t % 300 == 0:
        #     plot_map(mapData, t)

plot_map(mapData, len(R))


