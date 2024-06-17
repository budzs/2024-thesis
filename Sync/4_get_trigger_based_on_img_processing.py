import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
t = 0
source_directory = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\"
destination_directory = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\rgb"

# if not os.path.exists(destination_directory):
#      os.makedirs(destination_directory)

# for filename in os.listdir(source_directory):
#     if filename.endswith('.png'):
#         file_path = os.path.join(source_directory, filename)
#         image = Image.open(file_path)
        
#         if image.mode == 'RGB' and image.getbands() == ('R', 'G', 'B'):
#             shutil.copy(file_path, destination_directory)

directory = "C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\rgb"
# BGR values!!
lower_blue = np.array([180,180,0])
upper_blue = np.array([255,255,60])

files = os.listdir(directory)
files = sorted(files, key=lambda x: int(x.split('_')[1]))
kernel = np.ones((2,2),np.uint8)

with open("C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\trigger_frames.txt", "w") as f:
    f.write("Results\n")
    counter = 0
    list = []
    for i in range(0, len(files)-1):
        image_file = os.path.join(directory, files[i])
        frame = cv2.imread(os.path.join(directory, image_file))
        if frame is None:
            print(f"Error reading {image_file}")
            break
        height, width = frame.shape[:2]
        frame = frame[0:, :width//2]


        mask_blue = cv2.inRange(frame, lower_blue, upper_blue)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        blue_pixels = cv2.countNonZero(mask_blue)

        frame_number = image_file.split('_')[4]
        print(f"Frame {frame_number}: {blue_pixels} blue pixels")

        # cv2.imshow('Image', frame)
        # key = cv2.waitKey(5)  # Wait for a key press

        # if key == 13:  # 13 is the ASCII code for Enter
        #     print(f"{t}Image number: {frame_number}")
        #     t += 1

        if blue_pixels > 6:         
            if counter > 10:
                f.write(f"Frame {frame_number}: Detected\n")
                list.append(frame_number)
                counter = 0
    
        counter += 1
    print(list)

    plt.show()
    f.write("End of results\n")
    f.close()