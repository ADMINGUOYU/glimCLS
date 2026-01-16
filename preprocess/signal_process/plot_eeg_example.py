import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import signal

# Create output directory
save_dir = "eeg_figures_100hz"
os.makedirs(save_dir, exist_ok=True)

# Load the merged dataframe
with open('zuco_data/zuco_merged_with_variants.df', 'rb') as f:
    df = pickle.load(f)

print(f"Loaded dataframe with {len(df)} samples")

# Randomly sample 100 indices
np.random.seed(42)
sample_indices = np.random.choice(len(df), size=100, replace=False)

# Plot parameters
offset_scale = 15.0
max_freq_hz = 64  # Maximum frequency to display in frequency analysis plots

for idx in sample_indices:
    # Get EEG sample and use only channels 0-105
    eeg_sample = df.iloc[idx]['eeg'][:, :104]
    num_channels = eeg_sample.shape[1]

    # Get metadata
    sentiment = df.iloc[idx].get('sentiment label', 'N/A')
    dataset = df.iloc[idx].get('dataset', 'N/A')
    subject = df.iloc[idx].get('subject', 'N/A')

    # Create plot with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 12))
    time = range(len(eeg_sample))

    # Left subplot: Time-domain EEG
    for c in range(num_channels):
        ax1.plot(time, eeg_sample[:, c] + c * offset_scale, linewidth=0.6)

    ax1.set_title(f'EEG Time Domain - Sample {idx}', fontsize=12)
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Channel')

    yticks = [c * offset_scale for c in range(num_channels)]
    yticklabels = [f'Ch{c+1}' for c in range(num_channels)]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels, fontsize=6)
    ax1.invert_yaxis()

    # Right subplot: Frequency map (spectrogram of averaged channels)
    fs = 128  # Sampling rate in Hz
    avg_signal = np.mean(eeg_sample, axis=1)
    f, t, Sxx = signal.spectrogram(avg_signal, fs=fs, nperseg=128)

    im = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'Spectrogram (Averaged Channels)', fontsize=12)
    ax2.set_ylim([0, max_freq_hz])  # Focus on 0-max_freq_hz range for EEG
    plt.colorbar(im, ax=ax2, label='Power (dB)')

    # Third subplot: Frequency spectrum per channel (static map)
    fs = 128  # Sampling rate in Hz
    psd_matrix = []
    for c in range(num_channels):
        freqs, psd = signal.welch(eeg_sample[:, c], fs=fs, nperseg=256)
        psd_matrix.append(psd)

    psd_matrix = np.array(psd_matrix)
    freq_mask = freqs <= max_freq_hz  # Focus on 0-max_freq_hz

    im3 = ax3.pcolormesh(freqs[freq_mask], range(num_channels),
                         10 * np.log10(psd_matrix[:, freq_mask] + 1e-10),
                         shading='gouraud', cmap='jet')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Channel')
    ax3.set_title(f'Power Spectrum per Channel', fontsize=12)
    ax3.set_yticks([0, 25, 50, 75, 103])
    ax3.set_yticklabels(['Ch1', 'Ch26', 'Ch51', 'Ch76', 'Ch104'])
    ax3.invert_yaxis()
    plt.colorbar(im3, ax=ax3, label='Power (dB)')

    fig.suptitle(f'Sentiment: {sentiment} - {dataset} - Subject: {subject}', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'eeg_{idx:05d}.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)

    if (sample_indices.tolist().index(idx) + 1) % 10 == 0:
        print(f"Generated {sample_indices.tolist().index(idx) + 1}/100 figures")

print(f"All 100 figures saved to {save_dir}/")
