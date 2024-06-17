# %%
import sys

# %%
#!{sys.executable} -m pip install imbalanced-learn

# %%
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Dense, LSTM
import numpy as np

# Ensure compatibility with different TensorFlow versions
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# %%
def plot(loss, val_loss, acc, val_acc, euclidean_distance, val_euclidean_distance):
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, acc, 'b-', label='Training MAE')
    plt.plot(epochs, val_acc, 'r-', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, euclidean_distance, 'b-', label='Training Euclidean Distance')
    plt.plot(epochs, val_euclidean_distance, 'r-', label='Validation Euclidean Distance')
    plt.title('Training and Validation Euclidean Distance')
    plt.xlabel('Epochs')
    plt.ylabel('Distance')
    plt.legend()
    
    
    plt.tight_layout()
    plt.show()
# %%


def plot_training_history(history):
    plt.figure(figsize=(18, 6))

    # Plot for MSE
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss MSE')
    plt.title('Model Loss MSE')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Plot for MAE
    plt.subplot(1, 3, 2)
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

    # Plot for Euclidean Distance
    plt.subplot(1, 3, 3)
    if 'euclidean_distance' in history.history:
        plt.plot(history.history['euclidean_distance'], label='Train Euclidean Distance')
        plt.plot(history.history['val_euclidean_distance'], label='Validation Euclidean Distance')
        plt.title('Euclidean Distance')
        plt.ylabel('Distance')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# %%

data_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\processed_data_notch_norm\\'
class_dir = 'C:\\Users\\MindRove_BZs\\Diploma\\classes\\'

def load_data(patient_ids):
    emg_data = {}
    eeg_data = {}
    for i in patient_ids:
        with open(os.path.join(data_dir, f'normalized_filtered_emg_data_patient_{i}.pkl'), 'rb') as f:
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
    
def load_classes():
    class_files = [f for f in os.listdir(class_dir) if f.endswith('.pickle')]
    all_classes = {}
    for file in class_files:
        with open(os.path.join(class_dir, file), 'rb') as f:
            classes_data = pickle.load(f)
            identifier = file.split('_')[-1].replace('.pickle', '')
            for key in classes_data:
                modified_key = f"patient_{identifier}_{key}"
                all_classes[modified_key] = classes_data[key]
    return all_classes

def load_pca_data():
    base_path = "C:\\Users\\MindRove_BZs\\Diploma\\pca_labels"
    pca_data = {}
    for i in range(5, 12):
        file_path = os.path.join(base_path, f"{i}_pca_transformed_data.csv")
        with open(file_path, 'r') as f:
            identifier = file_path.split('_')[0].split('\\')[-1]
            #print(f"Identifier: {identifier}")
            for line in f:  
                parts = line.strip().split(',')  
                frame_number = parts[0]  
                #print(f"Frame number: {frame_number} ")
                data = [float(x) for x in parts[1:]] 
                modified_key = f"patient_{i}_{frame_number}" 
                #print(f"Modified key: {modified_key}")
                pca_data[modified_key] = data 

    return pca_data
            
# %%
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


def prepare_dataset(emg_data, eeg_data, all_classes):
    filtered_emg_data, filtered_eeg_data = filter_data(emg_data, eeg_data)
    X = []
    y = []
    for frame in all_classes.keys():
        #print(f"Frame: {frame}")
        if frame in filtered_emg_data and frame in filtered_eeg_data:
            combined_data = np.concatenate((filtered_emg_data[frame], filtered_eeg_data[frame]), axis=0)
            X.append(combined_data)
            y.append(all_classes[frame])
    X = np.array(X)
    y = np.array(y)
    return X, y



# %%
def euclidean_distance(y_true, y_pred):
    euclidean_distances = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
    return tf.reduce_mean(euclidean_distances)


def lstm_model(X):
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=128))
    model.add(Dense(units=60, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae', euclidean_distance])
    return model

# %%

def main():
    patient_ids = range(5, 12)
    emg_data, eeg_data = load_data(patient_ids)
    print(f"Number of frames in EMG data: {len(emg_data)}")
    print(f"Number of frames in EEG data: {len(eeg_data)}")
    
    all_classes = load_pca_data()
    print(f"Number of frames in all classes: {len(all_classes)}")    

    X, y = prepare_dataset(emg_data, eeg_data, all_classes)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    X_train = X_train.reshape((-1, 14, 30, 1))
    X_test = X_test.reshape((-1, 14, 30, 1))
    
    model = lstm_model(X_train)
    model.summary()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=100, verbose = 1)

    y_pred = model.predict(X_test)
    print("Predictions (subset):", y_pred[:10])
    print("Actual (subset):", y_test[:10])

    model.save('lstm_model.h5')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    euc_distance = euclidean_distance(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Euclidean Distance: {euc_distance}")
    plot(history.history['loss'], history.history['val_loss'], history.history.get('mae'), history.history.get('val_mae'), history.history.get('euclidean_distance'), history.history.get('val_euclidean_distance'))

    random_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)

    plt.figure(figsize=(10, 7))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(y_test[idx], label='Actual')
        plt.plot(y_pred[idx], label='Predicted', alpha=0.7)
        plt.title(f'Sample {idx+1}')
        plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(18, 6))

    plot_training_history(history)
main()



# %%
