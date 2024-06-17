import os
import shutil

dir_path = 'C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8'
files = os.listdir(dir_path)

for file in files:
    if file.endswith('.png'):
        timestamp = int(file.split('_')[2].split('.')[0])
        if timestamp > 1716462675665:
            os.remove(os.path.join(dir_path, file))