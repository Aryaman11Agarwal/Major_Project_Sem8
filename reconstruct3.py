import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===================== UNET BLOCKS (BIGGER) =====================

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv3D(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(32, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(64, 32)

        self.out_conv = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x = self.up1(x2)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        return self.out_conv(x)

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv2D(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv2D(32, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv2D(64, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x = self.up1(x2)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        return self.out_conv(x)

# ===================== FDK Placeholder =====================

def fdk_reconstruction(ap, lat):
    size = ap.shape[0]
    vol = np.zeros((size, size, size), dtype=np.float32)
    for i in range(size):
        vol[i] = (ap + lat.T) / 2
    return vol

# ===================== Dataset Class =====================

class PhantomDataset(Dataset):
    def __init__(self, npz_path):
        self.data = []
        npz_data = np.load(npz_path)
        keys = sorted(set(k.split('_')[1] for k in npz_data.files if k.startswith('volume')))
        for k in keys:
            vol = npz_data[f"volume_{k}"]
            ap = npz_data[f"ap_{k}"]
            lat = npz_data[f"lat_{k}"]
            self.data.append((vol, ap, lat))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vol, ap, lat = self.data[idx]
        vol = torch.tensor(vol[np.newaxis], dtype=torch.float32)
        ap = torch.tensor(ap, dtype=torch.float32)
        lat = torch.tensor(lat, dtype=torch.float32)
        return vol, ap, lat

# ===================== Training + Testing Loop =====================

def train_model(train_npz, test_npz, epochs=5, lr=1e-3, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    train_loader = DataLoader(PhantomDataset(train_npz), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PhantomDataset(test_npz), batch_size=1, shuffle=False)

    net3d = UNet3D().to(device)
    net2d = UNet2D().to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(list(net3d.parameters()) + list(net2d.parameters()), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for gt_vol, ap, lat in train_loader:
            gt_vol = gt_vol.to(device)
            ap = ap.to(device)
            lat = lat.to(device)

            fdk_vol = []
            for b in range(gt_vol.size(0)):
                fdk = fdk_reconstruction(ap[b].cpu().numpy(), lat[b].cpu().numpy())
                fdk_vol.append(fdk)
            fdk_vol = np.array(fdk_vol, dtype=np.float32)
            fdk_vol = torch.tensor(fdk_vol, dtype=torch.float32).unsqueeze(1).to(device)

            coarse_vol = net3d(fdk_vol)

            refined_slices = []
            for b in range(coarse_vol.size(0)):
                slices = []
                for i in range(coarse_vol.shape[2]):
                    slice_2d = coarse_vol[b, 0, i].unsqueeze(0).unsqueeze(0)
                    refined = net2d(slice_2d)
                    slices.append(refined.squeeze(0))
                refined_vol = torch.stack(slices, dim=0)
                refined_slices.append(refined_vol)
            output = torch.stack(refined_slices).unsqueeze(1).to(device)

            output = output.view_as(gt_vol)

            loss = criterion(output, gt_vol)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📘 Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.6f}")

    test_loss = 0.0
    final_gt = None
    final_output = None

    with torch.no_grad():
        for gt_vol, ap, lat in test_loader:
            gt_vol = gt_vol.to(device)
            ap = ap.to(device)
            lat = lat.to(device)

            fdk = fdk_reconstruction(ap[0].cpu().numpy(), lat[0].cpu().numpy())
            fdk = torch.tensor(fdk[np.newaxis, np.newaxis], dtype=torch.float32).to(device)

            coarse_vol = net3d(fdk)

            slices = []
            for i in range(coarse_vol.shape[2]):
                slice_2d = coarse_vol[0, 0, i].unsqueeze(0).unsqueeze(0)
                refined = net2d(slice_2d)
                slices.append(refined.squeeze(0))
            refined_vol = torch.stack(slices, dim=0).unsqueeze(0).unsqueeze(0).to(device)

            refined_vol = refined_vol.view_as(gt_vol)

            if final_gt is None:
                final_gt = gt_vol[0, 0].detach().cpu()
                final_output = refined_vol[0, 0].detach().cpu()

            loss = criterion(refined_vol, gt_vol)
            test_loss += loss.item()

    print(f"🧪 Test Loss: {test_loss / len(test_loader):.6f}")
    return final_gt, final_output

# ===================== MAIN =====================

if __name__ == "__main__":
    gt_volume, recon_volume = train_model("phantom_data.npz", "phantom_data_testing.npz", epochs=5)

    for i in range(10):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(gt_volume[i], cmap='gray')
        axs[0].set_title(f"GT Slice {i}")
        axs[0].axis('off')

        axs[1].imshow(recon_volume[i], cmap='gray')
        axs[1].set_title(f"Recon Slice {i}")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()
