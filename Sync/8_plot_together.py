import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
from scipy.signal import butter, iirnotch, filtfilt


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=1, fs=500, order=5): 
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def notch_filter(data, cutoff=50, q=30):  
    fs = 500
    normalized_cutoff = cutoff / (fs / 2)
    b, a = iirnotch(normalized_cutoff, q)
    y = filtfilt(b, a, data)
    return y

def display_data_eeg(start_image=100, num_images=8, image_directory="C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8", df_path="C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\landmarks.csv", eeg_df_path="C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\merged_data_frame_eeg.csv"):
    df = pd.read_csv(df_path)
    eeg_df = pd.read_csv(eeg_df_path)
    end_image = start_image + num_images

    images = sorted([img for img in os.listdir(image_directory) if img.endswith('.png')], key=lambda x: int(x.split('_')[1]))

    fig, axs = plt.subplots(4, num_images//2, figsize=(20, 15), gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1]})

    for ax in axs[2, :]:
        ax.remove()
    for ax in axs[3, :]:
        ax.remove()

    ax_big = plt.subplot2grid((4, num_images//2), (2, 0), colspan=num_images//2)
    ax_big_eeg = plt.subplot2grid((4, num_images//2), (3, 0), colspan=num_images//2, sharex=ax_big)
    plt.subplots_adjust(wspace=0, hspace=0.1)
    
    for i in range(start_image, end_image, 2):

        image_path = os.path.join(image_directory, images[i])
        image = mpimg.imread(image_path)
        row = 0
        col = (i - start_image) // 2  

        axs[row, col].set_xticks([]) 
        axs[row, col].set_yticks([]) 
        axs[row, col].imshow(image)
        axs[row, col].set_title(f'Image {i+1}', fontsize=6)
        axs[row, col].axis('off')

        image_path = os.path.join(image_directory, images[i+1])
        image = mpimg.imread(image_path)
        row = 1
        col = (i - start_image) // 2  

        axs[row, col].set_xticks([]) 
        axs[row, col].set_yticks([]) 
        cax = axs[row, col].imshow(image, vmin=0.0, vmax=0.01)
        axs[row, col].axis('off')
       
        frame_number = int(images[i].split('_')[1])
        for j in range(21):  
            landmark_x = df.loc[df['Frame'] == frame_number, f'landmark_{j}_x'].values[0]
            landmark_y = df.loc[df['Frame'] == frame_number, f'landmark_{j}_y'].values[0]
            axs[0, col].scatter(landmark_x, landmark_y, color='g', s=2, alpha=0.6)

    for channel in range(8):
        if channel == 6:
            continue
        eeg_data_all_frames = []
        for i in range(start_image, end_image + 1, 2):
            frame_number = int(images[i].split('_')[1])
            eeg_data = eeg_df.loc[eeg_df['Frame_EEG'] == frame_number, f'Channel_{channel}_EEG']
            eeg_data = notch_filter(eeg_data)
            eeg_data = highpass_filter(eeg_data) 
            eeg_data_all_frames.extend(eeg_data)
        ax_big.plot(eeg_data_all_frames, alpha=0.6, linewidth=0.5)
        ax_big.set_xlim(0, len(eeg_data_all_frames))
        ax_big.spines['top'].set_visible(False)
        ax_big.spines['right'].set_visible(False)
        ax_big.spines['bottom'].set_visible(False)
        ax_big.xaxis.set_visible(False)

    for channel in range(8):
        emg_data_all_frames = []
        for i in range(start_image, end_image + 1, 2):
            frame_number = int(images[i].split('_')[1])
            emg_data = eeg_df.loc[eeg_df['Frame'] == frame_number, f'Channel_{channel}_EMG']
            emg_data = notch_filter(emg_data)  
            emg_data = highpass_filter(emg_data) 
            emg_data_all_frames.extend(emg_data)
        ax_big_eeg.plot(emg_data_all_frames, alpha=0.6, linewidth=0.5)
        ax_big_eeg.set_xlim(0, len(emg_data_all_frames))
        ax_big_eeg.spines['top'].set_visible(False)
        ax_big_eeg.spines['right'].set_visible(False)
        
        
    plt.savefig('C:\\Users\\MindRove_BZs\\Pictures\\0523_patient_8\\plot.jpg', orientation='landscape', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


display_data_eeg(start_image=10011, num_images=32)
