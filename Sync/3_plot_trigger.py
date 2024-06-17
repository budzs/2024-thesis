import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

old_emg_csv_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 123]1.csv"
new_emg_csv_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 123]_emg.csv"

old_eeg_csv_path = 'C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 227]1.csv'
new_eeg_csv_path = 'C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 227]_eeg.csv'

if not os.path.exists(new_emg_csv_path):
    os.rename(old_emg_csv_path, new_emg_csv_path)
if not os.path.exists(new_eeg_csv_path):
    os.rename(old_eeg_csv_path, new_eeg_csv_path)

emg_csv_path = new_emg_csv_path
eeg_csv_path = new_eeg_csv_path
emg_df = pd.read_csv(emg_csv_path)
eeg_df = pd.read_csv(eeg_csv_path)

emg_df = emg_df.rename(columns={'trigger': 'Trigger'})
eeg_df = eeg_df.rename(columns={'trigger': 'Trigger'})
trigger_indices_emg = emg_df[emg_df['Trigger'] != 0].index
trigger_indices_eeg = eeg_df[eeg_df['Trigger'] != 0].index

data_points_between_triggers_emg = np.diff(trigger_indices_emg)
data_points_between_triggers_eeg = np.diff(trigger_indices_eeg)

print("EMG data points between triggers:", data_points_between_triggers_emg)
print("EEG data points between triggers:", data_points_between_triggers_eeg)

trigger_timestamps_emg = emg_df.loc[trigger_indices_emg, 'Timestamp'].reset_index(drop=True)
trigger_timestamps_eeg = eeg_df.loc[trigger_indices_eeg, 'Timestamp'].reset_index(drop=True)

for i, timestamp in enumerate(trigger_timestamps_eeg):
    if i < len(trigger_timestamps_emg):
        print(f"{i} EMG {timestamp}, EEG {trigger_timestamps_eeg[i]}")
    else:
        print(f"EEG {timestamp}")


print("Number of triggers in EMG data:", len(trigger_indices_emg))
print("Number of triggers in EEG data:", len(trigger_indices_eeg))

fig, axs = plt.subplots(1,2)

axs[0].plot(data_points_between_triggers_emg, 'x')

axs[0].set_ylim([4000, 11000])  
axs[0].set_title('EMG Data Points Between Triggers')  
axs[0].set_ylabel('Number of data points between triggers') 
axs[0].set_xlabel('Trigger Index')  

axs[1].plot(data_points_between_triggers_eeg, 'x')

axs[1].set_ylim([4000, 11000])  
axs[1].set_title('EEG Data Points Between Triggers')  
axs[1].set_ylabel('Number of data points between triggers')  
axs[1].set_xlabel('Trigger Index')  

plt.show()