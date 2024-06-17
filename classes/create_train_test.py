import os
import pickle
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers, Input, Model


data_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\processed_data\\'
class_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\classes\\'

def load_data(patient_ids):
    emg_data = {}
    eeg_data = {}
    for i in patient_ids:
        with open(os.path.join(data_dir, f'normalized_filtered_emg_data_patient_{i}.pkl'), 'rb') as f:
            emg_data.update(pickle.load(f))
        with open(os.path.join(data_dir, f'normalized_filtered_eeg_data_patient_{i}.pkl'), 'rb') as f:
            eeg_data.update(pickle.load(f))
    return emg_data, eeg_data

def load_classes():
    class_files = [f for f in os.listdir(class_dir) if f.endswith('.pickle')]
    all_classes = {}
    for file in class_files:
        with open(os.path.join(class_dir, file), 'rb') as f:
            all_classes.update(pickle.load(f))
    return all_classes

def filter_data(emg_data, eeg_data):
    filtered_emg_data = {}
    filtered_eeg_data = {}
    
    for frame in emg_data:
        if emg_data[frame].shape[1] >= 30 and eeg_data[frame].shape[1] >= 30:
            min_length = 30 #min(emg_data[frame].shape[1], eeg_data[frame].shape[1])
            filtered_emg_data[frame] = emg_data[frame][:-2, :min_length]
            filtered_eeg_data[frame] = eeg_data[frame][:-2, :min_length]
    
    return filtered_emg_data, filtered_eeg_data

def prepare_dataset(emg_data, eeg_data, all_classes):
    filtered_emg_data, filtered_eeg_data = filter_data(emg_data, eeg_data)
    X = []
    y = []
    for frame in all_classes.keys():
        if frame in filtered_emg_data and frame in filtered_eeg_data:
            #print(f"Frame: {frame}, EMG shape: {filtered_emg_data[frame].shape}, EEG shape: {filtered_eeg_data[frame].shape}")
            combined_data = np.concatenate((filtered_emg_data[frame], filtered_eeg_data[frame]), axis=0)
            #print(f"Combined data shape: {combined_data.shape}")       
            X.append(combined_data)
            #print(f"Shape of X: {np.array(X).shape}")
            y.append(all_classes[frame])
    X = np.array(X)
    y = to_categorical(np.array(y))  
    return X, y

def load_and_prepare():
    patient_ids = range(5, 8)  
    emg_data, eeg_data = load_data(patient_ids)
    print(f"Number of frames: {len(emg_data)}"
            f"\nNumber of EMG channels: {emg_data[list(emg_data.keys())[0]].shape[0]}"
            f"\nNumber of EEG channels: {eeg_data[list(eeg_data.keys())[0]].shape[0]}")
    all_classes = load_classes()
    print(f"Number of frames: {len(all_classes)}")
    print(f"Number of classes: {len(set(all_classes.values()))}")
    X, y = prepare_dataset(emg_data, eeg_data, all_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def model():
    input_shape = (12, 30, 1) 
    num_classes = 8  
    
    input_tensor = Input(shape=input_shape)
    
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(input_tensor, output_tensor)
    
    opt = optimizers.Nadam(learning_rate=1e-4)
    model.compile(optimizer=opt, 
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()
    return model

def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    
    X_train = X_train.reshape((-1, 12, 30, 1))
    X_test = X_test.reshape((-1, 12, 30, 1))
    
    cnn_model = model()

    history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
    cnn_model.save('C:\\Users\\MindRove_BZs\\Diploma\\models\\model.h5')

if __name__ == '__main__':
    main()