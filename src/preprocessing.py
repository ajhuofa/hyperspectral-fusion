import rasterio
import numpy as np
from scipy import ndimage

def load_and_preprocess(hsi_path, msi_path):
    with rasterio.open(hsi_path) as src:
        hsi = src.read()
        hsi_transform = src.transform
    
    with rasterio.open(msi_path) as src:
        msi = src.read()
        msi_transform = src.transform
    
    # Atmospheric correction and cloud masking
    hsi = atmospheric_correction(hsi)
    msi = atmospheric_correction(msi)
    
    # Co-registration
    msi_aligned = coregister_images(msi, msi_transform, hsi_transform)
    
    return hsi, msi_aligned

def atmospheric_correction(img):
    # Simple dark object subtraction
    return np.maximum(img - np.percentile(img, 1, axis=(1,2))[:, None, None], 0)

def coregister_images(img, src_transform, target_transform):
    # Basic image alignment using scipy
    return ndimage.affine_transform(img, src_transform, output_shape=target_transform)