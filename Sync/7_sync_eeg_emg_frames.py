import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\merged_data_frame.csv')

df['Frame_EEG'] = np.nan
trigger_frames = {}
for i in range(len(df)):
    if df.loc[i, 'Trigger_EMG'] != 0:  
        trigger_frames[df.loc[i, 'Trigger_EMG']] = df.loc[i, 'Frame']

for i in range(len(df)):
    if df.loc[i, 'Trigger_EEG'] in trigger_frames:
        df.loc[i, 'Frame_EEG'] = trigger_frames[df.loc[i, 'Trigger_EEG']]

df['Frame_EEG'] = df['Frame_EEG'].astype(float).interpolate(method='linear')
df['Frame_EEG'] = df['Frame_EEG'].apply(lambda x: int(round(x)) if int(round(x)) % 2 != 0 else int(round(x)) - 1)
df['Frame_EEG'] = df['Frame_EEG'].astype(int)
df.to_csv('C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\merged_data_frame_eeg.csv', index=False)