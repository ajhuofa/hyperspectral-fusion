import numpy as np
from osgeo import gdal

def load_image(filepath):
    """Load satellite image using GDAL"""
    dataset = gdal.Open(filepath)
    if dataset is None:
        raise ValueError(f'Unable to open {filepath}')
    return dataset.ReadAsArray()

def fusion_model(hyperspectral_img, high_res_img):
    """Basic fusion model implementation"""
    # TODO: Implement fusion algorithm
    pass