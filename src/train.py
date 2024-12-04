import torch
from torch.utils.data import DataLoader
from .fusion import FusionModule

def train_model(train_loader, val_loader, hsi_bands, msi_bands):
    model = FusionModule(hsi_bands, msi_bands)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    for epoch in range(100):
        model.train()
        for hsi, msi in train_loader:
            optimizer.zero_grad()
            output = model(hsi, msi)
            loss = criterion(output, hsi)
            loss.backward()
            optimizer.step()
    
    return model