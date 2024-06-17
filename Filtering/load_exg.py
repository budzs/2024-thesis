import pickle
import numpy as np  

total_frames = 0
dir = 'C:\\Users\\MindRove_BZs\\Diploma\\eeg_emg\\'

for i in range(5, 12):
    with open(dir + f'emg_data_patient_{i}.pkl', 'rb') as f:
        emg_data = pickle.load(f)
    print(f"Patient {i}: {len(emg_data)} frames for EMG data")
    print(f"Patient {i}: {len(emg_data[1001][0])} channels for EMG data")
    emg_frame_numbers = sorted([key for key in emg_data.keys() if not np.isnan(key)])
    print(f"Patient {i}: {len(emg_frame_numbers)} unique frames for EMG data")
    #print(f"EMG - First frame: {emg_frame_numbers[0]}, Last frame: {emg_frame_numbers[-1]}")

    with open(dir + f'eeg_data_patient_{i}.pkl', 'rb') as f:
        eeg_data = pickle.load(f)
    eeg_frame_numbers = sorted(eeg_data.keys())
    print(f"Patient {i}: {len(eeg_frame_numbers)} unique frames for EEG data")
    #print(f"EEG - First frame: {eeg_frame_numbers[0]}, Last frame: {eeg_frame_numbers[-1]}")

    total_frames += len(emg_frame_numbers)

print(f"Total number of unique frames: {total_frames}")