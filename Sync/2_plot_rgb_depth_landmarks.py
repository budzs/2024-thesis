import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def display_data(image_directory, df, start_image=100):
    base_path = "C:\\Users\\MindRove_BZs\\Pictures\\05232_patient_10\\"
    image_directory = [img for img in os.listdir(image_directory) if img.endswith('.png')]
    images = sorted(image_directory, key=lambda x: int(x.split('_')[1]))
    fig, axs = plt.subplots(3, 4, figsize=(20, 15)) 
    connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (5, 6), (6, 7), (7, 8),  # Index finger
                (9, 10), (10, 11), (11, 12),  # Middle finger
                (13, 14), (14, 15), (15, 16),  # Ring finger
                (0, 17), (17, 18), (18, 19), (19, 20),  # Little finger
                (2,5), (5, 9), (9, 13), (13, 17)]  # Palm 

    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(start_image, start_image+8):
        image_path_rgb = os.path.join(base_path, images[i])
        img = mpimg.imread(image_path_rgb)

        row = (i - start_image) % 2 # Determine the row for the image
        print(row)
        col = ((i - start_image) // 2) % 4  # Determine the column for the image

        if row == 0:  # RGB and landmarks
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
            for j in range(21):  
                frame_number = int(image_path_rgb.split('_')[4])
                landmark_x = df.loc[df['Frame'] == frame_number, f'landmark_{j}_x']
                landmark_y = df.loc[df['Frame'] == frame_number, f'landmark_{j}_y']
                axs[row, col].scatter(landmark_x, landmark_y, color='g', s=10)

            ax = fig.add_subplot(3, 4, 7 + col +2, projection='3d')            
            for j in range(21):  
                frame_number = int(image_path_rgb.split('_')[4])
                landmark_x = df.loc[df['Frame'] == frame_number, f'landmark_{j}_x']
                landmark_y = df.loc[df['Frame'] == frame_number, f'landmark_{j}_y']
                landmark_z = df.loc[df['Frame'] == frame_number, f'landmark_{j}_z']
                ax.scatter(landmark_x, landmark_y, landmark_z, color='g', s=10)

            ax.view_init(elev=-0., azim=-90)

            # Connect the landmarks
            for start, end in connections:
                frame_number = int(image_path_rgb.split('_')[4])
                start_x = df.loc[df['Frame'] == frame_number, f'landmark_{start}_x']
                start_y = df.loc[df['Frame'] == frame_number, f'landmark_{start}_y']
                start_z = df.loc[df['Frame'] == frame_number, f'landmark_{start}_z']
                end_x = df.loc[df['Frame'] == frame_number, f'landmark_{end}_x']
                end_y = df.loc[df['Frame'] == frame_number, f'landmark_{end}_y']
                end_z = df.loc[df['Frame'] == frame_number, f'landmark_{end}_z']
                ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], color='r')
                
        elif row == 1:  # Depth
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            axs[row, col].imshow(img,  vmin=0.00, vmax=0.05)
            axs[row, col].axis('off')

    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]:
        axs[r, c].axis('off')
    plt.show()

image_directory = "C:\\Users\\MindRove_BZs\\Pictures\\05232_patient_10\\"
df_path= "C:\\Users\\MindRove_BZs\\Pictures\\05232_patient_10\\landmarks1.csv"
df = pd.read_csv(df_path)

display_data(image_directory, df, start_image=1099)
