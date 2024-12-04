import numpy as np
from sklearn.metrics import mean_squared_error, peak_signal_noise_ratio

def evaluate_fusion(fused, reference):
    return {
        'rmse': np.sqrt(mean_squared_error(reference, fused)),
        'psnr': peak_signal_noise_ratio(reference, fused),
        'sam': spectral_angle_mapper(reference, fused)
    }

def spectral_angle_mapper(ref, test):
    return np.arccos(np.sum(ref * test) / 
                    (np.sqrt(np.sum(ref**2)) * np.sqrt(np.sum(test**2))))