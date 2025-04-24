import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from scipy.ndimage import rotate  # Correct 3D rotation
import random
import os

# ====== Create a 3D phantom with randomization and shape variation ======
def create_3d_phantom(slices=64, size=64, noise_level=0.05, rotate_angle_range=(-10, 10), body_part="generic"):
    # Base 2D phantom
    base = shepp_logan_phantom().astype(np.float32)
    base_resized = resize(base, (size, size), mode='reflect', anti_aliasing=True).astype(np.float32)

    # Add random noise
    noise = np.random.normal(scale=noise_level, size=base_resized.shape)
    base_resized += noise
    base_resized = np.clip(base_resized, 0, 1)

    # Create 3D volume by stacking
    volume = np.repeat(base_resized[np.newaxis, :, :], repeats=slices, axis=0)

    # Apply body-part-specific transformations
    if body_part == "head":
        volume = rotate(volume, angle=30, axes=(1, 2), reshape=False, mode='wrap')
    elif body_part == "chest":
        volume = np.clip(volume * 1.3, 0, 1)
    elif body_part == "abdomen":
        volume = np.clip(volume * 0.9, 0, 1)

    # Apply global 3D rotation for variability
    angle = random.uniform(*rotate_angle_range)
    volume = rotate(volume, angle=angle, axes=(0, 2), reshape=False, mode='wrap')

    # Add geometric shape: random sphere inside the volume
    if random.random() > 0.5:
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, slices),
            np.linspace(-1, 1, size),
            np.linspace(-1, 1, size),
            indexing='ij'
        )
        radius = random.uniform(0.2, 0.4)
        sphere = zz**2 + yy**2 + xx**2
        volume[sphere <= radius**2] += random.uniform(0.2, 0.5)
        volume = np.clip(volume, 0, 1)

    return volume.astype(np.float32)

# ====== Generate 2D Projections ======
def generate_projections(volume):
    ap_proj = np.mean(volume, axis=0).astype(np.float32)
    lat_proj = np.mean(volume, axis=2).astype(np.float32)
    return ap_proj, lat_proj

# ====== Display helper ======
def show_image(img, title=""):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# ====== Dataset Generator ======
def generate_dataset():
    try:
        num_images = int(input("Enter number of 3D phantom images to generate: "))
    except ValueError:
        print("Invalid input. Defaulting to 1 image.")
        num_images = 1

    size = 64
    slices = 64
    data_map = {}
    body_parts = ["generic", "head", "chest", "abdomen"]

    for i in range(num_images):
        body_part = random.choice(body_parts)
        volume = create_3d_phantom(slices, size, body_part=body_part)
        ap, lat = generate_projections(volume)
        data_map[i] = (volume, (ap, lat))
        print(f"✅ Created image {i + 1}/{num_images} — volume: {volume.shape}, body part: {body_part}")

    # Show first 10 samples
    for i in range(min(10, num_images)):
        show_image(data_map[i][0][slices // 2], f"Mid Slice of Image {i}")
        show_image(data_map[i][1][0], f"AP Projection {i}")
        show_image(data_map[i][1][1], f"LAT Projection {i}")

    return data_map

# ====== Save to .npz ======
def save_to_npz(data_map, filename="phantom_data.npz"):
    save_dict = {}
    for key, (volume, projections) in data_map.items():
        save_dict[f"volume_{key}"] = volume
        save_dict[f"ap_{key}"] = projections[0]
        save_dict[f"lat_{key}"] = projections[1]
    np.savez_compressed(filename, **save_dict)
    print(f"✅ Data saved to {filename}")

# ====== Run ======
if __name__ == "__main__":
    phantom_dataset = generate_dataset()
    save_to_npz(phantom_dataset, "phantom_data.npz")
