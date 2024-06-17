import csv
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import numpy as np

csv_directory = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8"
csv_file = os.path.join(csv_directory, 'landmarks.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

frames = []

min_x = min_y = min_z = np.inf
max_x = max_y = max_z = -np.inf

connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (5, 6), (6, 7), (7, 8),  # Index finger
                (9, 10), (10, 11), (11, 12),  # Middle finger
                (13, 14), (14, 15), (15, 16),  # Ring finger
                (0, 17), (17, 18), (18, 19), (19, 20),  # Little finger
                (2,5), (5, 9), (9, 13), (13, 17)]  # Palm 

with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        xs = [float(row[f'landmark_{i}_x']) for i in range(21)]
        ys = [float(row[f'landmark_{i}_y']) for i in range(21)]
        zs = [float(row[f'landmark_{i}_z']) for i in range(21)]

        min_x, max_x = min(min_x, min(xs)), max(max_x, max(xs))
        min_y, max_y = min(min_y, min(ys)), max(max_y, max(ys))
        min_z, max_z = min(min_z, min(zs)), max(max_z, max(zs))
        min_x, max_x = min_y, max_y = min(min_x, min_y), max(max_x, max_y)

        frames.append((xs, ys, zs))

def update(num, frames, ax):
    ax.clear()
    xs, ys, zs = frames[num]
    ax.scatter(xs, ys, zs, label=f'Frame {num}')
    for start, end in connections:
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'r')
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

ani = FuncAnimation(fig, update, frames=len(frames), fargs=[frames, ax], interval=10)
plt.show()