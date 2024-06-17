import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from angles import visualize_points
from angles_for_pcs import calculate_invariants

with open('C:\\Users\\MindRove_BZs\\Diploma\\modellek\\hand_model_all_55.pkl', 'rb') as f:
    saved_model = pickle.load(f)

pca = saved_model['pca']
norm_factors = saved_model['norm_factors']
pca_stats = saved_model['pca_stats']

def display_hand(landmarks, ax=None, title="Hand Landmarks"):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    visualize_points(landmarks, title, ax)
    ax.view_init(elev=90., azim=0)


def normalize_sample(sample, norm_factors):
    min_val = norm_factors['min_val']
    max_val = norm_factors['max_val']
    normalized_sample = (sample - min_val) / (max_val - min_val)
    return normalized_sample

def inverse_normalize_sample(normalized_sample, norm_factors):
    min_val = norm_factors['min_val']
    max_val = norm_factors['max_val']
    original_sample = normalized_sample * (max_val - min_val) + min_val
    return original_sample

def transform_with_pca(normalized_data, pca):
    print(f"Shape of the data before PCA transformation: {normalized_data.shape}")
    #reshaped_data = normalized_data.reshape(1, -1)  # Reshape for PCA transform
    reshaped_data = normalized_data
    pca_transformed_data = pca.transform(reshaped_data)
    print(f"Shape of the data after PCA transformation: {pca_transformed_data.shape}")
    return pca_transformed_data

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



def process_data_with_pca(file_path):
    df_points = pd.read_csv(file_path)
    frame_numbers = df_points['Frame'].values
    df_points = df_points.drop(columns=['Frame'])  
    new_data_landmarks = df_points.values.reshape(-1, 21, 3)  

    new_data = calculate_invariants(new_data_landmarks)
    reshaped_data = new_data.reshape(new_data.shape[0], -1)
    pca_transformed_new_data = transform_with_pca(reshaped_data, pca)

    data_dict = {frame: pca_data for frame, pca_data in zip(frame_numbers, pca_transformed_new_data)}
    df_pca = pd.DataFrame.from_dict(data_dict, orient='index')
    df_pca.to_csv('C:\\Users\\MindRove_BZs\\Pictures\\0507_patient_5\\pca_transformed_data55.csv', header=False)

    return pca_transformed_new_data, new_data_landmarks


def display_and_calculate_error(original_data, pca_transformed_data):
    reconstructed_data = pca.inverse_transform(pca_transformed_data)
    reconstructed_data = reconstructed_data.reshape(-1, 20, 3)
    reconstructed_landmarks = []
    for i in range(reconstructed_data.shape[0]):
        reconstructed_landmarks.append(reconstruct_hand_model(reconstructed_data[i]))

    # # Display the original and reconstructed landmarks
    # fig = plt.figure(figsize=(20, 10))
    # for i in range(len(reconstructed_landmarks)):
    #     plt.clf()  # Clear the current plot

    #     # Original landmarks
    #     ax1 = fig.add_subplot(121, projection='3d')
    #     # set axis limits
    #     ax1.set_xlim([300, 700])
    #     ax1.set_ylim([100, 500])
    #     display_hand(original_data[i], ax=ax1, title=f"Original Hand Landmarks {i}")

    #     # Reconstructed landmarks
    #     ax2 = fig.add_subplot(122, projection='3d')
    #     display_hand(reconstructed_landmarks[i], ax=ax2, title=f"Reconstructed Hand Landmarks {i}")

    #     plt.draw()  # Redraw the plot
    #     plt.pause(0.1)  # Pause to allow the plot to update

    # plt.show()  # Show the plot

    # Calculate the error between the original and reconstructed data
    error = np.mean(np.abs(original_data - reconstructed_landmarks))

    return error


file_path = "C:\\Users\\MindRove_BZs\\Pictures\\0507_patient_5\\landmarks1.csv"
pca_transformed_data, new_data_landmarks = process_data_with_pca(file_path)
error = display_and_calculate_error(new_data_landmarks, pca_transformed_data)
print(f"Error: {error}")

