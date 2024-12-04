# Hyperspectral-MSI Fusion

Transformer-based fusion of hyperspectral and multispectral imagery using spatial-spectral attention mechanisms.

## Features
- Dual attention mechanism (spatial and spectral)
- Preserves spectral information while enhancing spatial detail
- Supports synthetic training data generation

## Requirements
See requirements.txt

## Usage
```python
from src.fusion import FusionModule
model = FusionModule(hsi_bands=100, msi_bands=4)
```