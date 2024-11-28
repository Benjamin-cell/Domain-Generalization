import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

class MultiFeatureDataset(Dataset):
    def __init__(self, data, seq_len, short_term_len, predict_length):
        self.data = data
        self.seq_len = seq_len
        self.short_term_len = short_term_len
        self.predict_length = predict_length

    def __len__(self):
        return len(self.data) - self.seq_len - self.predict_length + 1

    def __getitem__(self, idx):
        x_long = self.data[idx:idx + self.seq_len]
        x_short = self.data[idx + self.seq_len - self.short_term_len:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.predict_length, -1]
        return torch.tensor(x_long, dtype=torch.float32), torch.tensor(x_short, dtype=torch.float32), torch.tensor(y,
                                                                                                                   dtype=torch.float32)

class InferenceDataset(Dataset):
    def __init__(self, data, seq_len, predict_length):
        self.data = data
        self.seq_len = seq_len
        self.predict_length = predict_length

    def __len__(self):
        return len(self.data) - self.seq_len - self.predict_length + 1

    def __getitem__(self, idx):
        x_long = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.predict_length, -1]
        return torch.tensor(x_long, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def split_few_shot_data(scaled_data):
    train_size = int(len(scaled_data) * 0.6)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    return train_data, test_data

def determine_window_lengths(time_series, time_step=1, min_short_window=24, max_long_window=30 * 24):
    import numpy as np
    from statsmodels.tsa.stattools import pacf
    from scipy.signal import welch
    N = len(time_series)
    # Step 1: 限制 PACF 的最大滞后期
    max_pacf_lag = min(40, N // 2)  # 设置最大滞后期为 40 或 N//2 中的较小值
    pacf_values = pacf(time_series, nlags=max_pacf_lag)
    conf_level = 1.96 / np.sqrt(N)
    significant_lags = np.where(np.abs(pacf_values) > conf_level)[0]
    # Step 2: 使用 Welch 方法估计功率谱密度（保持不变）
    frequencies, psd_values = welch(time_series, fs=1 / time_step, nperseg=min(256, N))
    peaks, _ = find_peaks(psd_values)
    peak_frequencies = frequencies[peaks]
    periods = 1 / peak_frequencies
    periods = periods[(periods > 0) & (periods < N)]
    # Step 3: 使用移动平均进行趋势分析
    window_size = min(max(5, int(0.1 * N)), N)  # 窗口大小为总长度的 10%，且不小于 5
    trend = np.convolve(time_series, np.ones(window_size) / window_size, mode='same')
    trend_change_time = estimate_trend_change_time_nonparametric(trend, time_step)
    # Step 4: 确定短期窗口（调整后）
    if len(significant_lags) > 1 and len(periods) > 0:
        short_term_window = max(min_short_window, min(significant_lags[1], int(min(periods) / (2 * time_step))))
    elif len(significant_lags) > 1:
        short_term_window = max(min_short_window, significant_lags[1])
    elif len(periods) > 0:
        short_term_window = max(min_short_window, int(min(periods) / (2 * time_step)))
    else:
        # 当没有显著滞后期或周期时，使用数据的自相关长度尺度
        autocorr_time = np.where(np.abs(pacf_values) < conf_level)[0]
        if len(autocorr_time) > 0:
            short_term_window = max(min_short_window, autocorr_time[0])
        else:
            short_term_window = min_short_window
    # Step 5: 确定长期窗口（保持不变）
    if len(periods) > 0:
        long_term_window = min(max_long_window, int(np.mean(periods[:3]) / time_step))
    else:
        # 使用趋势变化时间作为长期窗口的参考
        long_term_window = min(max_long_window, trend_change_time)
    return short_term_window, long_term_window
def estimate_trend_change_time_nonparametric(trend, time_step):
    # 使用趋势的一阶差分来估计趋势变化速率
    trend_diff = np.diff(trend)
    avg_change = np.mean(np.abs(trend_diff))
    if avg_change == 0:
        return len(trend)
    else:
        trend_range = np.max(trend) - np.min(trend)
        time_for_significant_change = (0.1 * trend_range) / avg_change
        return int(time_for_significant_change / time_step)