import pickle
import os

pickle_file_path = 'C:\\Users\\MindRove_BZs\\Diploma\\classes\\frame_class_dict_8.pickle'

if os.path.exists(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        frame_class_dict = pickle.load(f)
    
    for frame_number, class_number in frame_class_dict.items():
        print(f"Frame number: {frame_number}, Class: {class_number}")
else:
    print(f"File not found: {pickle_file_path}")