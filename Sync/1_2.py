import pandas as pd
import os

directory = "C:\\Users\\MindRove_BZs\\Pictures\\05233_patient_11"
merged_data_frame_eeg_df = pd.read_csv(os.path.join(directory, 'merged_data_frame_eeg.csv'))

for i in range(21):
    z_column = f'landmark_{i}_z'
    print(f"Number of missing values in {z_column}: {merged_data_frame_eeg_df[z_column].isnull().sum()}")
    merged_data_frame_eeg_df[z_column] = merged_data_frame_eeg_df[z_column].replace(0, pd.NA)
    merged_data_frame_eeg_df[z_column].fillna(method='ffill', inplace=True)
    

merged_data_frame_eeg_df.to_csv(os.path.join(directory, 'merged_data_frame_eeg1.csv'), index=False)