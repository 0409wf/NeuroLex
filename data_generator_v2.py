# data_generator_v2.py
import numpy as np


def generate_eeg_sample_v2(is_known, length=256, n_channels=3, noise_level=0.3, p300_strength=1.2):
    t = np.linspace(0, 1, length)
    sample = []

    for _ in range(n_channels):
        signal = 0.4 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.sin(2 * np.pi * 20 * t)
        signal += noise_level * np.random.randn(length)  # 加强噪声

        if is_known and np.random.rand() > 0.1:  # 假阴性10%
            # P300模拟，峰值带随机性
            peak_time = np.random.uniform(0.25, 0.4)
            p300 = np.exp(-((t - peak_time) ** 2) / 0.002)
            signal += p300_strength * p300
        elif not is_known and np.random.rand() < 0.1:  # 假阳性10%
            # 假冒的“错觉记住”
            peak_time = np.random.uniform(0.25, 0.4)
            p300 = np.exp(-((t - peak_time) ** 2) / 0.002)
            signal += 0.5 * p300

        sample.append(signal)

    return np.array(sample)  # shape: (n_channels, length)


def generate_dataset_v2(n_samples=200):
    X, y = [], []
    for _ in range(n_samples):
        is_known = np.random.rand() > 0.5
        x = generate_eeg_sample_v2(is_known)
        X.append(x)
        y.append(1 if is_known else 0)
    return np.array(X), np.array(y)  # shape: (N, C, T), label: 0 or 1
