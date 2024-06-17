import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt

from scipy.signal import freqz
import matplotlib.pyplot as plt

fs = 500 
cutoff = 2 
order = 4  

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=4):
    #print(f"len data {len(data)}")
    if len(data) <= 15:
        print("Data length is too short for filtering. Skipping filter.")
        return data 
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)  
    return y

def normalize_data(data):
    normalized_data = {}
    emg_frame_numbers = sorted([key for key in data.keys() if not np.isnan(key)])
    print(f"Found {len(emg_frame_numbers)} EMG frames")
    print()

    channels = [[] for _ in range(8)]  

    for key in emg_frame_numbers:
        for ch in range(8):  
            channels[ch].extend(data[key][ch])

    for ch in range(8):
        min_val = min(channels[ch])
        max_val = max(channels[ch])
        normalized_values = [(value - min_val) / (max_val - min_val) if max_val > min_val else 0 for value in channels[ch]]

        index = 0
        for key in emg_frame_numbers:
            frame_length = len(data[key][ch])
            normalized_data.setdefault(key, [[] for _ in range(8)])
            normalized_data[key][ch] = normalized_values[index:index+frame_length]
            index += frame_length      

    return normalized_data


fs = 500  
t = np.arange(0, 1.0, 1/fs) 
f = 5  
square_wave = np.sign(np.sin(2 * np.pi * f * t))
square_wave = square_wave + 10
# notch filter
f0 = 50  
Q = 10  
b_notch, a_notch = iirnotch(f0, Q, fs)

def apply_notch_filter(data, b_notch, a_notch):
    filtered_data = filtfilt(b_notch, a_notch, data, axis=0)
    return filtered_data

data = {0: [square_wave for _ in range(8)]}

#normalized_data = normalize_data(data)
filtered_data = {}
for key, channels in data.items():
    filtered_channels = []
    for channel in channels:
        channel_notched = apply_notch_filter(channel, b_notch, a_notch)
        filtered_channel = highpass_filter(channel_notched, cutoff, fs, order)
        filtered_channels.append(filtered_channel)
    filtered_data[key] = filtered_channels

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, square_wave, label='Original Square Wave')
plt.title('Original Square Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, filtered_data[0][0], label='Filtered Square Wave')
plt.title('Filtered Square Wave (One Channel Example)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()


b, a = butter_highpass(cutoff, fs, order=order)

w, h = freqz(b, a, worN=8000)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("High-pass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()