import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import signal

def robust_normalize_padded(eeg_data, axis=-1, clip_value=10.0, epsilon=1e-6):
    """
    对含有 Zero-Padding 的定长 EEG 片段进行鲁棒标准化。
    1. 识别并“掩盖”掉填充的 0 值。
    2. 仅利用有效数据（非 0 部分）计算均值和标准差。
    3. 执行标准化 (Z-score)。
    4. 将原来的填充位置重新填回 0。
    
    Args:
        eeg_data (np.ndarray): 输入 EEG 信号，形状通常为 (Channels, 1280)。
                               假设尾部含有补齐用的 0 值。
        axis (int): 时间轴，通常为 -1。
        clip_value (float): 截断阈值，限制输出范围。
        epsilon (float): 防止除以零的极小值。
    
    Returns:
        np.ndarray: 形状保持不变 (Channels, 1280)，但有效部分已标准化，
                    无效部分保持为 0。
    """
    # 1. 创建掩码数组 (Masked Array)
    masked_data = np.ma.masked_equal(eeg_data, 0)
    # print 0元素的个数
    # print("Number of masked (zero) elements:", np.sum(masked_data.mask))

    # 2. 计算均值和标准差
    # numpy.ma 的函数会自动忽略被 mask 的元素
    # keepdims=True 保证维度不被压缩，方便后续广播计算
    mean = np.mean(masked_data, axis=axis, keepdims=True)
    std = np.std(masked_data, axis=axis, keepdims=True)
    
    # 3. 执行 Z-score 标准化
    # 计算只会在未被 mask 的数据上进行，mask 的位置结果依然是 mask
    normalized_masked = (masked_data - mean) / (std + epsilon)
    
    # 4. 截断 (Clipping)
    if clip_value is not None:
        # ma.clip 会自动处理掩码结构
        normalized_masked = np.ma.clip(normalized_masked, -clip_value, clip_value)
    
    # 5. 填回 0 值 (Fill zeros)
    # 将被 mask 的位置（原本是 0 的位置）重新填充为 0
    # .data 返回底层的 numpy 数组
    final_data = normalized_masked.filled(0)
    
    return final_data


def spectral_whitening(eeg_data, alpha=0.95):
    """
    对 EEG 信号进行频谱白化（预加重），以此抵消 1/f 噪声，
    增强高频（Gamma）信号的显著性。
    
    公式: y[t] = x[t] - alpha * x[t-1]
    
    Args:
        eeg_data (np.ndarray): 输入 EEG 信号。
                               建议形状: (n_channels, n_timepoints) 
                               或者 (n_batch, n_channels, n_timepoints)
        alpha (float): 预加重系数。通常取 0.95 或 0.97。
                       alpha=0 不改变信号，alpha=1 近似于一阶差分。
    
    Returns:
        np.ndarray: 白化后的 EEG 信号，形状与输入相同。
    """
    # 确保输入是浮点型，防止溢出
    eeg_data = eeg_data.astype(np.float32)
    
    # 初始化输出数组
    whitened_data = np.zeros_like(eeg_data)
    
    # 处理 t=0 时刻 (没有 t-1，保持原值)
    # 也可以选择填充 0，但保留原值能减少边缘突变
    whitened_data[..., 0] = eeg_data[..., 0]
    
    # 处理 t > 0 时刻: x[t] - alpha * x[t-1]
    # 使用切片操作进行向量化加速，适用于任何维度，只要最后一个维度是时间
    whitened_data[..., 1:] = eeg_data[..., 1:] - alpha * eeg_data[..., :-1]
    
    return whitened_data



def plot_eeg(eeg_sample, save_dir,label=None):
    eeg_sample=eeg_sample.T
    # Plot parameters
    offset_scale = 15.0
    max_freq_hz = 64  # Maximum frequency to display in frequency analysis plots
    num_channels = eeg_sample.shape[1]

    # Create plot with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 12))
    time = range(len(eeg_sample))

    # Left subplot: Time-domain EEG
    for c in range(num_channels):
        ax1.plot(time, eeg_sample[:, c] + c * offset_scale, linewidth=0.6)

    ax1.set_title(f'EEG Time Domain', fontsize=12)
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

    # fig.suptitle(f'', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'eeg_{label}.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)


