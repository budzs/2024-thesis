from tensorflow.keras.models import load_model
import os
import glob
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt



def load_data(patient_ids, data_dir='C:\\Users\\MindRove_BZs\\Diploma\\processed_data_notch_norm\\'):
    emg_data = {}
    eeg_data = {}
    for i in patient_ids:
        filename_print = f'normalized_filtered_emg_data_patient_{i}.pkl'
        with open(os.path.join(data_dir, filename_print), 'rb') as f:
            patient_emg_data = pickle.load(f)
            for key in patient_emg_data:
                if isinstance(key, float) and key.is_integer():
                    key = int(key)  
                modified_key = f"patient_{i}_{key}"
                emg_data[modified_key] = patient_emg_data[key]
        
        with open(os.path.join(data_dir, f'normalized_filtered_eeg_data_patient_{i}.pkl'), 'rb') as f:
            patient_eeg_data = pickle.load(f)
            for key in patient_eeg_data:
                if isinstance(key, float) and key.is_integer():
                    key = int(key) 
                modified_key = f"patient_{i}_{key}"
                eeg_data[modified_key] = patient_eeg_data[key]
        print(f"Size of eeg data after patient {i}: {len(eeg_data)}")
    
    return emg_data, eeg_data

def filter_data(emg_data, eeg_data):
    channels_to_keep = np.r_[0:6, 7:8] 

    filtered_emg_data = {}
    filtered_eeg_data = {}
    for frame in emg_data:
        if emg_data[frame].shape[1] >= 30 and eeg_data[frame].shape[1] >= 30:
            min_length = 30
            filtered_emg_data[frame] = emg_data[frame][:, :min_length]#[channels_to_keep][:, :min_length]
            filtered_eeg_data[frame] = eeg_data[frame][channels_to_keep][:, :min_length]
    return filtered_emg_data, filtered_eeg_data


def prepare_dataset(emg_data, eeg_data, all_classes):
    filtered_emg_data, filtered_eeg_data = filter_data(emg_data, eeg_data)
    
    print(f"Number of frames in EMG data: {len(emg_data)}")
    print(f"Number of frames in EEG data: {len(eeg_data)}")
    print(f"Number of frames in all classes: {len(all_classes)}")
    X_dict = {}
    y_dict = {}

    for frame in all_classes.keys():
        #print(f"FRAME {frame}")
        if frame in filtered_emg_data and frame in filtered_eeg_data:
            combined_data = np.concatenate((filtered_emg_data[frame], filtered_eeg_data[frame]), axis=0)
            X_dict[frame] = filtered_emg_data[frame] 
            y_dict[frame] = all_classes[frame]

    return X_dict, y_dict


def load_classes(class_dir='C:\\Users\\MindRove_BZs\\Diploma\\classes\\', filename='frame_class_dict_7.pickle'):
    file_path = os.path.join(class_dir, filename)
    print(f"Opening file {filename}")

    with open(file_path, 'rb') as f:
        classes_data = pickle.load(f)
        identifier = filename.split('_')[-1].replace('.pickle', '')
        all_classes = {}
        for key in classes_data:
            modified_key = f"patient_{identifier}_{key}"
            all_classes[modified_key] = classes_data[key]
    return all_classes



def visualize_predictions(X_test, y_test, model, images, all_classes):
    probabilities = {f"Class {i}": [] for i in range(8)}
    keys = []
    true_labels = []

    for i, frame_key in enumerate(all_classes.keys()):

        if frame_key in X_test.keys() and frame_key in y_test.keys():
            keys.append(frame_key)
            test_x = X_test.get(frame_key)
            test_y = y_test.get(frame_key)
            true_labels.append(test_y)
            test_x = np.expand_dims(test_x, axis=0) 
            test_x = np.expand_dims(test_x, axis=-1) 
            prediction = model.predict(test_x)
            for i, prob in enumerate(prediction[0]):
                probabilities[f"Class {i}"].append(prob)
        else:
            continue

    for i in range(8):
        plt.figure(figsize=(10, 8))  

        plt.subplot(2, 1, 1) 
        prob_to_plot = probabilities[f"Class {i}"][1000:2000]
        plt.plot(range(len(prob_to_plot)), prob_to_plot, linestyle='-')
        plt.title(f"Class {i} Probabilities")
        plt.xlabel("Frame")
        plt.ylabel("Probability")

        plt.subplot(2, 1, 2)  
        true_labels_binary = [1 if label == i else 0 for label in true_labels[1000:2000]]
        plt.plot(range(len(true_labels_binary)), true_labels_binary, linestyle='-', marker='o', markersize=5)
        plt.title(f"True Labels for Class {i}")
        plt.xlabel("Frame")
        plt.ylabel("Is True Label")

        plt.tight_layout() 
        plt.show()


def load_images(image_dir):
    images = {}
    for img_path in glob.glob(os.path.join(image_dir, '*.png')):
        #print(img_path)
        identifier = os.path.basename(img_path).split('_')[1]
        identifier = f"patient_7_{identifier}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[identifier] = img
        #print(f"Identifier {identifier}")
    return images

def main():
    data_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\processed_data\\'
    class_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\classes\\'
    model_filepath = "C:\\Users\\MindRove_BZs\\Downloads\\model_results.h5"#'C:\\Users\\MindRove_BZs\\Diploma\\models\\81_model_results.h5'
    image_dir = 'C:\\Users\\MindRove_BZs\\Pictures\\0522_patient_7\\rgb\\'
    images = load_images(image_dir)
    patient_ids = range(7, 8)
    emg_data, eeg_data = load_data(patient_ids)
    print(f"Number of frames in EMG data: {len(emg_data)}")
    print(f"Number of frames in EEG data: {len(eeg_data)}")
    all_classes = load_classes()
    print(f"Number of frames in all classes: {len(all_classes)}")    
    X, y = prepare_dataset(emg_data, eeg_data, all_classes)
    model = load_model(model_filepath)
    
    visualize_predictions(X, y, model, images, all_classes)

main()