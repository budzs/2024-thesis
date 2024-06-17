import pandas as pd

df = pd.read_csv("C:\\Users\\MindRove_BZs\\Pictures\\05233_patient_11\\merged_data_frame_eeg1.csv")

emg_data = {}
eeg_data = {}

for index, row in df.iterrows():
    print(f"Processing row {index}...")
    emg_frame = row['Frame']
    eeg_frame = row['Frame_EEG']
    
    emg_row_data = [row['Channel_0_EMG'], row['Channel_1_EMG'], row['Channel_2_EMG'], row['Channel_3_EMG'], row['Channel_4_EMG'], row['Channel_5_EMG'], row['Channel_6_EMG'], row['Channel_7_EMG']]
    eeg_row_data = [row['Channel_0_EEG'], row['Channel_1_EEG'], row['Channel_2_EEG'], row['Channel_3_EEG'], row['Channel_4_EEG'], row['Channel_5_EEG'], row['Channel_6_EEG'], row['Channel_7_EEG']]
    
    if emg_frame not in emg_data:
        emg_data[emg_frame] = [[] for _ in range(8)]
    if eeg_frame not in eeg_data:
        eeg_data[eeg_frame] = [[] for _ in range(8)]
    
    for i in range(8):
        emg_data[emg_frame][i].append(emg_row_data[i])
        eeg_data[eeg_frame][i].append(eeg_row_data[i])

import pickle

with open('emg_data_patient_11.pkl', 'wb') as f:
    pickle.dump(emg_data, f)

with open('eeg_data_patient_11.pkl', 'wb') as f:
    pickle.dump(eeg_data, f)

frame = 1001
print("EMG data for frame", frame)
print(emg_data[frame])
print("EEG data for frame", frame)
print(eeg_data[frame])