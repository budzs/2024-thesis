import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.spatial.transform import Rotation as R


file_path = "C:\\Users\\MindRove_BZs\\Pictures\\05233_patient_11\\landmarks1.csv"
pca_file_path = "C:\\Users\\MindRove_BZs\\Pictures\\05233_patient_11\\pca_transformed_data.csv"
data_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\processed_data_notch_norm\\'
model_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\models\\lstm_model.h5'

with open('C:\\Users\\MindRove_BZs\\Diploma\\modellek\\hand_model_all_60.pkl', 'rb') as f:
    saved_model = pickle.load(f)

pca = saved_model['pca']
norm_factors = saved_model['norm_factors']
pca_stats = saved_model['pca_stats']


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
    print(f"Shape: {all_return_arr.shape}")

    return all_return_arr

def load_data(patient_ids = [11]):
    emg_data = {}
    eeg_data = {}
    for i in patient_ids:
        with open(os.path.join(data_dir, f'normalized_filtered_emg_data_patient_{i}.pkl'), 'rb') as f:
            patient_emg_data = pickle.load(f)
            for key in patient_emg_data:
                if isinstance(key, float) and key.is_integer():
                    key = int(key)  
                modified_key = f"patient_{i}_{key}"
                emg_data[key] = patient_emg_data[key]
        
        with open(os.path.join(data_dir, f'normalized_filtered_eeg_data_patient_{i}.pkl'), 'rb') as f:
            patient_eeg_data = pickle.load(f)
            for key in patient_eeg_data:
                if isinstance(key, float) and key.is_integer():
                    key = int(key)  
                modified_key = f"patient_{i}_{key}"
                eeg_data[key] = patient_eeg_data[key]
        print(f"Size of eeg data after patient {i}: {len(eeg_data)}")
    
    return emg_data, eeg_data


def display_hand(landmarks, ax=None, title="Hand Landmarks"):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    visualize_points(landmarks, title, ax)
    ax.view_init(elev=90., azim=0)


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
    
def reconstruct_hand_model(data):
    azimuths = np.concatenate(([0], data[:, 0]))
    elevations = np.concatenate(([0], data[:, 1]))
    distances = np.concatenate(([0], data[:, 2]))

    ref_distance = distances[1]
    points = [np.zeros(3)] 

    for i in range(1, len(distances)):
        azimuth = azimuths[i]  
        elevation = elevations[i] 
        distance = distances[i] * ref_distance  
        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)

        points.append(np.array([x, y, z]))

    return np.array(points)

def display(original_data, reconstructed_landmarks, predicted_landmarks):

    fig = plt.figure(figsize=(20, 10))
    for i in range(len(reconstructed_landmarks)):
        plt.clf()  

        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_xlim([0, 500])
        ax1.set_ylim([0, 500])
        display_hand(original_data[i], ax=ax1, title=f"Original Hand Landmarks {i}")

        ax2 = fig.add_subplot(132, projection='3d')
        display_hand(reconstructed_landmarks[i], ax=ax2, title=f"Reconstructed Hand Landmarks {i}")

        ax3 = fig.add_subplot(133,projection='3d' )
        display_hand(predicted_landmarks[i], ax=ax3, title=f"Predicted Hand Landmarks {i}")

        plt.draw() 
        plt.pause(0.1) 
        plt.waitforbuttonpress()
      
    plt.show() 
    error = np.mean(np.abs(original_data - reconstructed_landmarks))

    return error

def filter_data(emg_data, eeg_data):
    channels_to_keep = np.r_[0:6, 7:8]

    filtered_emg_data = {}
    filtered_eeg_data = {}
    for frame in emg_data:
        if emg_data[frame].shape[1] >= 30 and eeg_data[frame].shape[1] >= 30:
            min_length = 30
            filtered_emg_data[frame] = emg_data[frame][channels_to_keep][:, :min_length]
            filtered_eeg_data[frame] = eeg_data[frame][channels_to_keep][:, :min_length]
    return filtered_emg_data, filtered_eeg_data

def main():
  emg_data, eeg_data = load_data([11])
  filtered_emg_data, filtered_eeg_data = filter_data(emg_data, eeg_data)
  print(f"filtered eeg {len(filtered_emg_data.keys())}")
  df_points = pd.read_csv(file_path)
  frame_numbers = df_points['Frame'].values

  original_data_filtered = df_points[df_points['Frame'].isin(filtered_emg_data.keys())]
  print(f"Len of og: {len(original_data_filtered)}")
  df_points = original_data_filtered.drop(columns=['Frame'])
  original_data = df_points.values.reshape(-1, 21, 3) 

  pca_points = pd.read_csv(pca_file_path)
  pca_points_filtered = pca_points[pca_points.iloc[:, 0].isin(filtered_emg_data.keys())]
  print(f"Len of pca: {len(pca_points_filtered)}")
  pca_points = pca_points_filtered.drop(pca_points.columns[0], axis=1)
  pca_points = pca_points.values.reshape(-1, 20, 3)

  pca_transformed_data_reshaped = pca_points.reshape(-1, 60)
  print("before inverse pca")
  reconstructed_data = pca.inverse_transform(pca_transformed_data_reshaped)
  print("after inverse pca")
  reconstructed_data = reconstructed_data.reshape(-1, 20, 3)

  reconstructed_landmarks = []
  for i in range(reconstructed_data.shape[0]):
      reconstructed_landmarks.append(reconstruct_hand_model(reconstructed_data[i]))

  model = load_model(model_dir)
  model_outputs = []
  common_frames = set(filtered_emg_data.keys()) & set(filtered_eeg_data.keys())
  i = 0
  for frame in common_frames:
    print(f"Out of {len(common_frames)} got: {i}")
    i+= 1

    combined_data = np.concatenate((filtered_emg_data[frame], filtered_eeg_data[frame]), axis=0)
    combined_data = combined_data.reshape((-1, 14, 30, 1))
    prediction = model.predict(combined_data)

    prediction = prediction.reshape(-1, 20, 3)
    for pred in prediction:
        model_outputs.append(pred)
  predicted_landmarks = []

  for i in range(len(model_outputs)):
    predicted_landmarks.append(reconstruct_hand_model(model_outputs[i]))

  display(original_data, reconstructed_landmarks, predicted_landmarks )

if __name__ == "__main__":
    
    main()