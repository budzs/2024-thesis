import pandas as pd

trigger_frames = ['221', '519', '817', '1115', '1413', '1711', '2009', 
                  '2307', '2605', '2903', '3201', '3499', '3797', '4095', 
                  '4393', '4691', '4989', '5287', '5585', '5883', '6181', 

                  
                  '6777', '7075', '7373', '7671', '7969', '8267', '8565', 
                  '8863', '9161', '9459', '9757', '10055', '10353', '10651', '10949', 

                  '11545', '11843', '12141', '12439', '12737', '13035', '13333', 
                  '13631', '13929', '14227', '14525', '14823', '15107',

                  '15407', '15705', '16003', 

                  '16593', '16889', 

                  '17485', '17783', '18081', '18379', '18677', '18975', 
                  '19273', '19571', '19869', '20167', '20465', '20763', 
                  '21061']


print(len(trigger_frames))  
emg_csv_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 123]_emg.csv"
frame_emg_csv_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 123]_emg_frame.csv"
emg_df = pd.read_csv(emg_csv_path)
landmarks_csv_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\landmarks.csv" 
landmarks_df = pd.read_csv(landmarks_csv_path)

emg_df = emg_df.rename(columns={'trigger': 'Trigger'})
emg_df['Frame'] = None  

first_trigger_index = emg_df[emg_df['Trigger'] == 2].index[0]
last_trigger_index = emg_df[emg_df['Trigger'] == 2].index[-1]

emg_df = emg_df.loc[first_trigger_index:last_trigger_index]
print(f"Number of rows in EMG: {len(emg_df)}")

trigger_indices = emg_df[emg_df['Trigger'] == 2].index.tolist() 

for i, trigger_index in enumerate(trigger_indices):
    emg_df.at[trigger_index, 'Frame'] = int(trigger_frames[i])  

emg_df['Frame'] = emg_df['Frame'].astype(float).interpolate(method='linear')

emg_df['Frame'] = emg_df['Frame'].apply(lambda x: int(round(x)) if int(round(x)) % 2 != 0 else int(round(x)) - 1)
emg_df['Frame'] = emg_df['Frame'].astype(int)

merged_df = pd.merge(emg_df, landmarks_df, on='Frame', how='left')
print(f"Number of rows in merged: {len(merged_df)}")


merged_df.to_csv(frame_emg_csv_path, index=False)