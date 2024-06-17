import pickle
from collections import Counter

pickle_files = ['frame_class_dict_11.pickle', 'frame_class_dict_9.pickle', 
                'frame_class_dict_10.pickle', 'frame_class_dict_5.pickle', 'frame_class_dict_6.pickle', 
                'frame_class_dict_7.pickle', 'frame_class_dict_8.pickle']

pickle_files = ['frame_class_dict_8.pickle']
base_path = 'C:\\Users\\MindRove_BZs\\Diploma\\classes'
all_class_counts = Counter()

for pickle_file in pickle_files:
    pickle_file = base_path + '\\' + pickle_file
    with open(pickle_file, 'rb') as f:
        frame_class_dict = pickle.load(f)

    class_counts = Counter(frame_class_dict.values())
    all_class_counts += class_counts

    print(f"\nClass representation in {pickle_file}:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count}")

print("\nTotal class representation:")
for class_label, count in all_class_counts.items():
    print(f"Class {class_label}: {count}")
