from matplotlib import pyplot as plt

def plot_frames(original_data, normalized_data, title, num_frames=10, num_channels=8):
    keys = list(original_data.keys())[:num_frames]  
    num_plots = len(keys) * (num_channels - 1) 
    
    fig, axs = plt.subplots(num_plots, 2, figsize=(15, 2*num_plots))
    fig.suptitle(title)
    
    for i, key in enumerate(keys):
        for ch in range(num_channels):
            if ch == 6: 
                continue
            idx = i * (num_channels - 1) + ch
            if ch > 6: 
                idx -= 1
            axs[idx, 0].plot(original_data[key][:, ch])
            if ch == 0:
                axs[idx, 0].set_title(f'Frame {key} - Before Normalization')
            axs[idx, 0].set_ylabel(f'Ch {ch+1}')
            axs[idx, 1].plot(normalized_data[key][:, ch])
            if ch == 0:
                axs[idx, 1].set_title(f'Frame {key} - After Normalization')
            axs[idx, 1].set_ylabel(f'Ch {ch+1}')
    
    for ax in axs.flat:
        ax.set_xlabel('Sample')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_combined_frames_with_normalized(original_data, normalized_data, title, num_frames=500, num_channels=8):
    keys = list(original_data.keys())[200:num_frames] 
    global_min_origin = float('inf')
    global_max_origin = float('-inf')
    global_min_filtered = float('inf')
    global_max_filtered = float('-inf')
    
    for ch in range(num_channels):
        if ch == 6: 
            continue
        for key in keys:
            global_min_origin = min(global_min_origin, min(original_data[key][:, ch]), min(normalized_data[key][:, ch]))
            global_max_origin = max(global_max_origin, max(original_data[key][:, ch]), max(normalized_data[key][:, ch]))
            global_min_filtered = min(global_min_filtered, min(normalized_data[key][:, ch]))
            global_max_filtered = max(global_max_filtered, max(normalized_data[key][:, ch]))
    
    fig, axs = plt.subplots(num_channels - 1, 2, figsize=(30, 2*(num_channels - 1)), sharex='col')
    fig.suptitle(title)
    
    for ch in range(num_channels):
        if ch == 6:  
            continue
        adjusted_ch = ch if ch < 6 else ch - 1  
        
        combined_data_original = []
        combined_data_normalized = []
        for key in keys:
            combined_data_original.extend(original_data[key][:, ch])
            combined_data_normalized.extend(normalized_data[key][:, ch])
        
        axs[adjusted_ch, 0].plot(combined_data_original, linewidth=0.5)
        axs[adjusted_ch, 0].set_ylabel(f'Ch {ch+1}')
        axs[adjusted_ch, 0].set_title(f'Original Channel {ch+1}', loc='left')
        
        axs[adjusted_ch, 1].plot(combined_data_normalized, linewidth=0.5)
        axs[adjusted_ch, 1].set_title(f'Filtered and normalized Channel {ch+1}', loc='left')
        axs[adjusted_ch, 1].set_ylim(global_min_filtered, global_max_filtered)
    
    axs[-1, 0].set_xlabel('Sample')
    axs[-1, 1].set_xlabel('Sample')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()