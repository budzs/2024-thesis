import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import os

def load_and_display_image(frame_number, class_number, base_folder):
    image_pattern = os.path.join(base_folder, f"frame_{frame_number}_*.png")
    matching_files = glob.glob(image_pattern)
    
    if matching_files:
        image_path = matching_files[0]
        img = plt.imread(image_path)
        plt.figure()
        plt.imshow(img)
        plt.title(f"Frame: {frame_number}, Class: {class_number}")
        plt.axis('off') 
        plt.show()
    else:
        print(f"Image for frame {frame_number} not found.")

def classify_landmarks(row, previous_row):
    # Extract the z-coordinates of the landmarks
    z_coords = row[2::3]
    #previous_z_coords = previous_row[2::3] if previous_row is not None else [0]*21

    # Check if the fingers are raised together
    finger_tips_z = [z_coords.iloc[i] for i in [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]]
    finger_tips_z_4 = [z_coords.iloc[i] for i in [ 4, 8, 12, 16, 20]]
    
    # Check if the hand is clenched (ököl)
    finger_tips = [row[i*3:i*3+3] for i in [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]]
    wrist = row[0:3]
    distances = [np.linalg.norm(np.array(finger_tip) - np.array(wrist)) for finger_tip in finger_tips]
    print(f"mean distances: {np.mean(distances)}")
    
    if np.mean(distances) < 0.58:
        print("Class 5")
        return 5
    if np.mean(finger_tips_z) < 0.55:
        print("Class 6")
        return 6

    # Check if a single finger tip is significantly closer to the camera than the others
    if z_coords[4] < np.mean(finger_tips_z) - np.std(finger_tips_z):
        print("Class 0")
        return 0
    elif  z_coords[8] < np.mean(finger_tips_z) - np.std(finger_tips_z)/2:
        print("Class 1")
        return 1
    elif z_coords[12] < np.mean(finger_tips_z) - np.std(finger_tips_z)/2:
        print("Class 2")
        return 2
    elif  z_coords[16] < np.mean(finger_tips_z_4) - np.std(finger_tips_z_4):
        print("Class 3")
        return 3
    elif z_coords[20] < np.mean(finger_tips_z) - np.std(finger_tips_z):
        print("Class 4")
        return 4
    print("Class 7")
    return 7

def process_data(file_path, patient_id, start_frame=1, end_frame=10, base_folder=""):
    df_points = pd.read_csv(file_path)
    frame_numbers = df_points['Frame'].values  
    df_points = df_points.drop(columns=['Frame']) 

    df_points_all = df_points.astype(float)
    for i in range(3):  
        min_val = df_points_all.iloc[:, i::3].min().min()
        max_val = df_points_all.iloc[:, i::3].max().max()
        df_points_all.iloc[:, i::3] = (df_points_all.iloc[:, i::3] - min_val) / (max_val - min_val)


    df_points = df_points_all
    #df_points = df_points_all.iloc[start_frame-1:end_frame]
    new_data_landmarks = df_points.values.reshape(-1, 21, 3) 

    df_points['class_label'] = df_points.apply(lambda row: classify_landmarks(row, df_points.iloc[df_points.index.get_loc(row.name)-1] if df_points.index.get_loc(row.name) > 0 else None), axis=1)
    labeled_data = np.c_[new_data_landmarks.reshape(new_data_landmarks.shape[0], -1), df_points['class_label'].values]
    frame_class_dict = {frame: class_label for frame, class_label in zip(frame_numbers, df_points['class_label'].values)}

    base_path = 'C:\\Users\\MindRove_BZs\\Diploma\\classes'
    name = os.path.join(base_path, f'frame_class_dict_{patient_id}.pickle')
    with open(name, 'wb') as f:
        pickle.dump(frame_class_dict, f)

    random_frames = [167, 227, 275, 319, 409, 435, 509, 555, 601, 647]
 
    for frame in random_frames:
        class_number = frame_class_dict.get(frame)
        if class_number is not None:
            load_and_display_image(frame, class_number, base_folder)
        else:
            print(f"Class number for frame {frame} not found.")


    return labeled_data, new_data_landmarks

file_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\landmarks1.csv"
base_folder = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\rgb"
labeled_data, new_data_landmarks = process_data(file_path, 8, 1305, 1399, base_folder)
