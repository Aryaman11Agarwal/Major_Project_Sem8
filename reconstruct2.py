import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

# ===================== CNN MODELS =====================

class Simple3DNet(nn.Module):
    def __init__(self):
        super(Simple3DNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Simple2DNet(nn.Module):
    def __init__(self):
        super(Simple2DNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# ===================== Realistic FDK via Radon Transform =====================



def fdk_reconstruction(ap, lat, angles=None):
    if angles is None:
        angles = np.linspace(0., 180., max(ap.shape), endpoint=False)

    # Reconstruct 2D slices from AP and LAT using inverse Radon
    slices = []
    for i in range(ap.shape[0]):
        # Treat each row of AP as a sinogram
        sinogram = np.tile(ap[i], (len(angles), 1)).T
        recon = iradon(sinogram, theta=angles, circle=True)
        slices.append(recon)
    volume = np.stack(slices, axis=0)

    # Average with reconstruction from LAT for more realism
    slices_lat = []
    for i in range(lat.shape[1]):
        sinogram = np.tile(lat[:, i], (len(angles), 1)).T
        recon = iradon(sinogram, theta=angles, circle=True)
        slices_lat.append(recon)
    volume_lat = np.stack(slices_lat, axis=2)

    volume_combined = (volume + volume_lat) / 2.0
    return volume_combined.astype(np.float32)

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

# ===================== Training Function =====================

def train_model(npz_file, epochs=5, lr=1e-3, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    dataset = PhantomDataset(npz_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net3d = Simple3DNet().to(device)
    net2d = Simple2DNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(net3d.parameters()) + list(net2d.parameters()), lr=lr)

    final_gt = None
    final_output = None

    for epoch in range(epochs):
        total_loss = 0.0
        for gt_vol, ap, lat in loader:
            gt_vol = gt_vol.to(device)
            ap = ap.to(device)
            lat = lat.to(device)

            fdk_vol = []
            for b in range(gt_vol.size(0)):
                fdk = fdk_reconstruction1(ap[b].cpu().numpy(), lat[b].cpu().numpy())
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

            final_gt = gt_vol[0, 0].detach().cpu()
            final_output = output[0, 0].detach().cpu()

        print(f"📘 Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.6f}")

    return final_gt, final_output

# ===================== MAIN =====================

if __name__ == "__main__":
    gt_volume, recon_volume = train_model("phantom_data.npz", epochs=5)

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
