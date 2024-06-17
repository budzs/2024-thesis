import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  


def load_dataframes(emg_csv_path, eeg_csv_path):
    emg_df = pd.read_csv(emg_csv_path)
    eeg_df = pd.read_csv(eeg_csv_path)

    emg_df = emg_df.rename(columns={'trigger': 'Trigger'})
    eeg_df = eeg_df.rename(columns={'trigger': 'Trigger'})

    previous_trigger = 0
    previous_trigger_index = 0

    for i in range(0, len(eeg_df)):
        temp = eeg_df.loc[i, 'Trigger']
        if temp != 0:
            if(i - previous_trigger_index > 8000):
                ind = (i + previous_trigger_index) // 2
                eeg_df.loc[ind, 'Trigger'] = previous_trigger + 1
                previous_trigger = previous_trigger + 1
                eeg_df.loc[i, 'Trigger'] =  previous_trigger + 1
                previous_trigger = previous_trigger + 1
            else:
                previous_trigger = previous_trigger + 1
                eeg_df.loc[i, 'Trigger'] = previous_trigger

            previous_trigger_index = i
    print(f"Number of triggers in EEG: {previous_trigger}")
    # eeg_df.to_csv('C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 227]_eeg_m.csv', index=False)

    previous_trigger = 0
    previous_trigger_index = 0
    
    for i in range(0, len(emg_df)):
        temp = emg_df.loc[i, 'Trigger']
        if temp != 0:
            if(i - previous_trigger_index > 8000):
                ind = (i + previous_trigger_index) // 2
                emg_df.loc[ind, 'Trigger'] = previous_trigger + 1
                previous_trigger = previous_trigger + 1
                emg_df.loc[i, 'Trigger'] =  previous_trigger + 1
                previous_trigger = previous_trigger + 1
            else:
                previous_trigger = previous_trigger + 1
                emg_df.loc[i, 'Trigger'] = previous_trigger

            previous_trigger_index = i
    print(f"Number of triggers in EMG: {previous_trigger}")

    # emg_df.to_csv('C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 51]_emg_m.csv', index=False)

    return emg_df, eeg_df

def find_first_last_trigger(df):
    return df[df['Trigger'] != 0]['Trigger'].first_valid_index(), df[df['Trigger'] != 0]['Trigger'].last_valid_index()

def slice_and_reset_indices(df, first_trigger, last_trigger):
    df = df.loc[first_trigger:last_trigger]
    df = df.reset_index(drop=True)
    return df

def merge_dataframes(emg_df, eeg_df, name = 'merged_data_frame.csv'):
    merged_df = pd.merge(emg_df, eeg_df, left_index=True, right_index=True, suffixes=('_EMG', '_EEG'), how='outer')
    merged_df.to_csv('C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\' + name, index=False)

def get_slice(df, start_trigger, end_trigger):
    start_index = df[df['Trigger'] == start_trigger].first_valid_index()
    end_index = df[df['Trigger'] == end_trigger].last_valid_index()
    df = df.loc[start_index:end_index]
    df = df.reset_index(drop=True)
    return df


def main():
    emg_csv_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 123]_emg_frame.csv"
    eeg_csv_path = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\csv_data_[10, 42, 0, 227]_eeg.csv"

    emg_df, eeg_df = load_dataframes(emg_csv_path, eeg_csv_path)

    first_trigger_emg, last_trigger_emg = find_first_last_trigger(emg_df)
    first_trigger_eeg, last_trigger_eeg = find_first_last_trigger(eeg_df)

    emg_df = slice_and_reset_indices(emg_df, first_trigger_emg, last_trigger_emg)
    eeg_df = slice_and_reset_indices(eeg_df, first_trigger_eeg, last_trigger_eeg)

    merge_dataframes(emg_df, eeg_df)

    # emg_df_1 = get_slice(emg_df, 1, 2)
    # eeg_df_1 = get_slice(eeg_df, 1, 2)
    # merge_dataframes(emg_df_1, eeg_df_1, 'merged_data_frame.csv')
       


if __name__ == "__main__":
    main()