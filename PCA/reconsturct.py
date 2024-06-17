import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def rotate_to_align(points, p1_idx=0, p2_idx=11):
    p1 = points[p1_idx]
    p2 = points[p2_idx]
    vec = p2 - p1
    vec /= np.linalg.norm(vec)

    target_vec = np.array([0, 1, 0])
    axis = np.cross(vec, target_vec)
    angle = np.arccos(np.dot(vec, target_vec))
    axis /= np.linalg.norm(axis)

    rotation = R.from_rotvec(axis * angle).as_matrix()
    rotated_points = np.dot(points - p1, rotation) + p1

    return rotated_points

def reconstruct_points(distances, azimuths, elevations):
    ref_distance = distances[1]
    points = [np.zeros(3)]
    for i in range(1, len(distances)):
        distance = distances[i] * ref_distance
        azimuth = azimuths[i-1]
        elevation = elevations[i-1]

        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)

        points.append(np.array([x, y, z]))

    return np.array(points)

def main():
    dir = 'C:\\Users\\MindRove_BZs\\Pictures\\0507_patient_5'
    df = pd.read_csv(f'{dir}\\updated_landmarks.csv')
    reconstructed_points = []
    for _, row in df.iterrows():
        distances = row[1:21].values  
        azimuths = row[21:41].values  
        elevations = row[41:].values 

        points = reconstruct_points(distances, azimuths, elevations)
        points = rotate_to_align(points)
        reconstructed_points.append(points.flatten()) 

    points_df = pd.DataFrame(reconstructed_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    connections = [[0, 1], [1, 2], [2, 3],  # Thumb
                [4, 5], [5, 6], [6, 7],  # Index finger
                [8, 9], [9, 10], [10, 11],  # Ring finger
                [12, 13],  [13, 14], [14, 15],  # Middle
                [16, 17], [17, 18], [18, 19],  # Pinky
                [0, 16], [2, 4],[4, 8], [8, 12], [12, 16],  # Palm
                ] 

    for _, row in points_df.iterrows():
        xs, ys, zs = row.values.reshape(-1, 3).T  
        ax.scatter(xs, ys, zs)
    
        for start, end in connections:
             ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'b-')
    
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
    
        plt.draw()
        plt.pause(0.1) 
        ax.clear()

    plt.show()

if __name__ == '__main__':
    main()