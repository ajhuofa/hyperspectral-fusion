import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Generate spatial attention weights
        weights = self.conv(torch.mean(x, dim=1, keepdim=True))
        return x * weights

class SpectralAttention(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.band_mix = nn.Conv2d(bands, bands, kernel_size=1)
        self.weights = nn.Sigmoid()
    
    def forward(self, x):
        # Generate band-wise importance weights
        weights = self.weights(self.band_mix(x))
        return x * weights

class FusionModule(nn.Module):
    def __init__(self, hsi_bands, msi_bands):
        super().__init__()
        self.spatial_att = SpatialAttention()
        self.spectral_att = SpectralAttention(hsi_bands)
        self.final_conv = nn.Conv2d(hsi_bands + msi_bands, hsi_bands, 1)
    
    def forward(self, hsi, msi):
        # Apply attention mechanisms
        hsi_feat = self.spatial_att(hsi)
        hsi_feat = self.spectral_att(hsi_feat)
        
        # Upsample MSI to match HSI spatial dimensions
        msi_up = torch.nn.functional.interpolate(
            msi, size=hsi_feat.shape[-2:], mode='bilinear'
        )
        
        # Concatenate and fuse
        fused = torch.cat([hsi_feat, msi_up], dim=1)
        return self.final_conv(fused)