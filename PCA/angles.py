import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def random_rotation(points):
    rotation = R.from_euler('xyz', np.random.uniform(0, 360, size=3), degrees=True).as_matrix()
    rotated_points = np.dot(points, rotation)
    return rotated_points

def zoom_points(points, factor):
    return points * factor

def rotate_to_align(points, p1_idx=0, p2_idx=11):
    points = points.astype(float)  
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

def calculate_invariants(all_points):
    all_return_arr = []

    for points in all_points:
        rotated_points = rotate_to_align(points)
        points = rotated_points

        origin = points[0]
        x_axis = points[1] - origin
        x_axis /= np.linalg.norm(x_axis)

        temp_vec = points[2] - origin
        z_axis = np.cross(x_axis, temp_vec)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis])

        centered_points = points - origin
        local_points = np.dot(centered_points, rotation_matrix.T)

        ref_distance = np.linalg.norm(local_points[1])

        distances = np.linalg.norm(centered_points[1:], axis=1) / ref_distance
        azimuths = []
        elevations = []

        for i in range(1, len(local_points)):
            x, y, z = local_points[i]
            azimuth = np.arctan2(y, x)
            elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
            azimuths.append(azimuth)
            elevations.append(elevation)

        azimuths = np.array(azimuths)[..., np.newaxis]
        elevations = np.array(elevations)[..., np.newaxis]
        distances = distances[..., np.newaxis]

        return_arr = np.concatenate([azimuths, elevations, distances], axis=-1)
        all_return_arr.append(return_arr)

    all_return_arr = np.array(all_return_arr)
    #print(f"Shape: {all_return_arr.shape}")

    return all_return_arr

def reconstruct_hand_model(distances, azimuths, elevations):
    distances = np.concatenate(([0], distances))
    azimuths = np.concatenate(([0], azimuths))
    elevations = np.concatenate(([0], elevations))
    ref_distance = distances[1]
    points = [np.zeros(3)]  

    for i in range(1, len(distances)):
        distance = distances[i] * ref_distance
        azimuth = azimuths[i]
        elevation = elevations[i]

        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)

        points.append(np.array([x, y, z]))
    print(f"Points shape: {np.array(points).shape}")

    return np.array(points)

def visualize_points(points, title, ax):
    connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (5, 6), (6, 7), (7, 8),  # Index finger
                (9, 10), (10, 11), (11, 12),  # Middle finger
                (13, 14), (14, 15), (15, 16),  # Ring finger
                (0, 17), (17, 18), (18, 19),(19,20),  # Little finger
                (2,5), (5, 9), (9, 13), (13, 17)]  # Palm 

    ax.scatter(points[:,0], points[:,1], points[:,2])
    for connection in connections:
         p1 = points[connection[0]]
         p2 = points[connection[1]]
         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def compare_invariants(invariants1, invariants2, atol=1e-6):
    distances_match = np.allclose(invariants1[0], invariants2[0], atol=atol)
    azimuths_match = np.allclose(invariants1[1], invariants2[1], atol=atol)
    elevations_match = np.allclose(invariants1[2], invariants2[2], atol=atol)
    print("Distances match:", distances_match)
    print("Azimuths match:", azimuths_match)
    print("Elevations match:", elevations_match)
    return distances_match and azimuths_match and elevations_match


def main():
    directory = 'C:\\Users\\MindRove_BZs\\Pictures\\0517_patient_6'
    df_points = pd.read_csv(f'{directory}\\landmarks1.csv')
    points_list = []
    fig = plt.figure(figsize=(16, 12))  
    ax1 = fig.add_subplot(1, 2, 1, projection='3d') 
    ax2 = fig.add_subplot(1, 2, 2, projection='3d') 

    for _, row in df_points.iterrows():
        points_list.append(row[1:].values.reshape(21, 3))

    for i, points in enumerate(points_list):
        ax1.cla()
        ax2.cla()  

        visualize_points(points, f"Original Points {i}", ax1)
        points = points.reshape(1, 21, 3)
        invariants  = calculate_invariants(points)
        distances, azimuths, elevations = invariants[0][:, 2], invariants[0][:, 0], invariants[0][:, 1]
        reconstructed_points = reconstruct_hand_model(distances, azimuths, elevations)
        visualize_points(reconstructed_points, "Reconstructed Points", ax2)

        plt.draw()
        plt.pause(0.05) 
        plt.title(f"Frame {i}")


if __name__ == '__main__':
    main()