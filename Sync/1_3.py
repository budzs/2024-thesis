import pandas as pd
import os
import time 

dir = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8"
df = pd.read_csv(os.path.join(dir, 'csv_data_[10, 42, 0, 227].csv'))

i = 0
while i < len(df):
    print(f"{df.loc[i, 'Timestamp']} not between 1716461964904 and 1716462675625")
    if not 1716461964904 <= df.loc[i, 'Timestamp'] <= 1716462675665:
        df = df.drop(i)
        df = df.reset_index(drop=True)
    else:
        i += 1

df.to_csv(os.path.join(dir, 'csv_data_[10, 42, 0, 227]1.csv'), index=False)