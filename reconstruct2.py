import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

# ===================== Deep UNet3D =====================
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.ReLU())

        self.middle = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1), nn.ReLU()
        )

        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv3d(512, 256, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv3d(256, 128, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(96, 64, 3, padding=1), nn.ReLU())

        self.final = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        m = self.middle(e4)

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)

# ===================== Deep UNet2D =====================
class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())

        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(96, 64, 3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        m = self.middle(e4)

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)

# ===================== FDK via Radon/iradon =====================
def fdk_reconstruction(ap, lat, angles=None):
    

    volume_ap = []
    for i in range(ap.shape[0]):
        sinogram = np.stack([ap[i], np.flip(ap[i])], axis=0)
        recon = iradon(sinogram.T, theta=[0, 180], circle=True)
        volume_ap.append(recon)
    volume_ap = np.stack(volume_ap, axis=0)

    volume_lat = []
    for i in range(ap.shape[1]):
        sinogram = np.stack([lat[:, i], np.flip(lat[:, i])], axis=0)
        recon = iradon(sinogram.T, theta=[0, 180], circle=True)
        volume_lat.append(recon)
    volume_lat = np.stack(volume_lat, axis=2)

    return ((volume_ap + volume_lat) / 2.0).astype(np.float32)

# ===================== Dataset =====================
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
        return (
            torch.tensor(vol[np.newaxis], dtype=torch.float32),
            torch.tensor(ap, dtype=torch.float32),
            torch.tensor(lat, dtype=torch.float32)
        )

# ===================== Training & Testing =====================
def train_and_test(train_npz, test_npz, epochs=5, lr=1e-3, batch_size=1):
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
            fdk_vol = torch.tensor(np.array(fdk_vol), dtype=torch.float32).unsqueeze(1).to(device)

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

            loss = criterion(output.view_as(gt_vol), gt_vol)
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

            loss = criterion(refined_vol.view_as(gt_vol), gt_vol)
            test_loss += loss.item()

            if final_gt is None:
                final_gt = gt_vol[0, 0].detach().cpu()
                final_output = refined_vol[0, 0].detach().cpu()

    print(f"🧪 Test Loss: {test_loss / len(test_loader):.6f}")
    return final_gt, final_output

# ===================== MAIN =====================
if __name__ == "__main__":
    gt_volume, recon_volume = train_and_test("phantom_data.npz", "phantom_data_testing.npz", epochs=5)

    for i in range(10):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(gt_volume[i], cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"GT Slice {i}")
        axs[0].axis('off')
        axs[1].imshow(recon_volume[i], cmap='gray', vmin=0, vmax=1)
        axs[1].set_title(f"Recon Slice {i}")
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()
