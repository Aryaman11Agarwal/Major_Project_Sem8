import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
import random

def create_3d_phantom(slices=64, size=64, rotate_angle_range=(-10, 10), body_part="generic"):
    volume = np.zeros((slices, size, size), dtype=np.float32)
    used_mask = np.zeros_like(volume, dtype=bool)
    possible_intensities = [0.3, 0.5, 0.7]

    def place_shape(mask, intensity):
        if np.any(np.logical_and(mask, used_mask)):
            return False
        volume[mask] = intensity
        used_mask[mask] = True
        return True

    def add_sphere():
        zz, yy, xx = np.meshgrid(np.arange(slices), np.arange(size), np.arange(size), indexing='ij')
        for _ in range(10):
            center = np.array([random.randint(10, slices-10), random.randint(10, size-10), random.randint(10, size-10)])
            radius = random.randint(5, 10)
            mask = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2 <= radius**2
            intensity = random.choice(possible_intensities)
            if place_shape(mask, intensity):
                return

    def add_ellipsoid():
        zz, yy, xx = np.meshgrid(np.linspace(0, slices-1, slices),
                                 np.linspace(0, size-1, size),
                                 np.linspace(0, size-1, size),
                                 indexing='ij')
        for _ in range(10):
            cz, cy, cx = [random.randint(10, dim-10) for dim in [slices, size, size]]
            rz, ry, rx = [random.randint(5, 12) for _ in range(3)]
            mask = ((zz-cz)/rz)**2 + ((yy-cy)/ry)**2 + ((xx-cx)/rx)**2 <= 1
            intensity = random.choice(possible_intensities)
            if place_shape(mask, intensity):
                return

    def add_cube():
        for _ in range(10):
            z0, y0, x0 = random.randint(0, slices-10), random.randint(0, size-10), random.randint(0, size-10)
            d = random.randint(5, 10)
            mask = np.zeros_like(volume, dtype=bool)
            mask[z0:z0+d, y0:y0+d, x0:x0+d] = True
            intensity = random.choice(possible_intensities)
            if place_shape(mask, intensity):
                return

    def add_cuboid():
        for _ in range(10):
            z0, y0, x0 = random.randint(0, slices-10), random.randint(0, size-10), random.randint(0, size-10)
            dz, dy, dx = random.randint(3, 8), random.randint(5, 12), random.randint(5, 12)
            mask = np.zeros_like(volume, dtype=bool)
            mask[z0:z0+dz, y0:y0+dy, x0:x0+dx] = True
            intensity = random.choice(possible_intensities)
            if place_shape(mask, intensity):
                return

    def add_cylinder():
        for _ in range(10):
            center_y = random.randint(10, size-10)
            center_x = random.randint(10, size-10)
            radius = random.randint(5, 8)
            height = random.randint(10, slices//2)
            z_start = random.randint(0, slices-height)

            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            circle_mask = (yy - center_y)**2 + (xx - center_x)**2 <= radius**2

            mask = np.zeros_like(volume, dtype=bool)
            for z in range(z_start, z_start + height):
                mask[z][circle_mask] = True

            intensity = random.choice(possible_intensities)
            if place_shape(mask, intensity):
                return

    def add_defect():
        zz, yy, xx = np.meshgrid(np.arange(slices), np.arange(size), np.arange(size), indexing='ij')
        center = np.array([random.randint(5, slices-5), random.randint(5, size-5), random.randint(5, size-5)])
        radius = random.randint(2, 5)
        mask = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2 <= radius**2
        volume[mask] = 0
        used_mask[mask] = False

    shape_fns = [add_sphere, add_ellipsoid, add_cube, add_cuboid, add_cylinder]
    random.shuffle(shape_fns)
    shape_count = 0
    while shape_count < 10:
        shape_fn = random.choice(shape_fns)
        prev_mask = used_mask.copy()
        shape_fn()
        if np.count_nonzero(used_mask) > np.count_nonzero(prev_mask):
            shape_count += 1

    for _ in range(random.randint(1, 3)):
        add_defect()

    if body_part == "head":
        volume = rotate(volume, angle=30, axes=(1, 2), reshape=False, mode='constant', cval=0)
    elif body_part == "chest":
        volume = np.clip(volume * 1.3, 0, 1)
    elif body_part == "abdomen":
        volume = np.clip(volume * 0.9, 0, 1)

    angle = random.uniform(*rotate_angle_range)
    volume = rotate(volume, angle=angle, axes=(0, 2), reshape=False, mode='constant', cval=0)

    return volume.astype(np.float32)

# ====== Generate 2D Projections with enhancement ======
def generate_projections(volume):
    ap_proj = np.mean(volume, axis=0)
    lat_proj = np.mean(volume, axis=2)

    # Optional enhancement
    ap_proj = gaussian_filter(ap_proj, sigma=0.5)
    lat_proj = gaussian_filter(lat_proj, sigma=0.5)

    # Normalize to [0, 1]
    ap_proj = (ap_proj - ap_proj.min()) / (ap_proj.max() - ap_proj.min() + 1e-8)
    lat_proj = (lat_proj - lat_proj.min()) / (lat_proj.max() - lat_proj.min() + 1e-8)

    return ap_proj.astype(np.float32), lat_proj.astype(np.float32)

# ====== Display helper ======
def show_image(img, title=""):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)  # Fixed intensity scale
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
        print(f"\u2705 Created image {i + 1}/{num_images} — volume: {volume.shape}, body part: {body_part}")

    if num_images > 0:
        first_volume, (ap_proj, lat_proj) = data_map[0]
        for slice_idx in range(min(10, slices)):
            show_image(first_volume[slice_idx], f"Slice {slice_idx + 1} of Image 0")
        show_image(ap_proj, "AP Projection of Image 0")
        show_image(lat_proj, "LAT Projection of Image 0")

    return data_map

# ====== Save to .npz ======
def save_to_npz(data_map, filename="phantom_data_testing.npz"):
    save_dict = {}
    for key, (volume, projections) in data_map.items():
        save_dict[f"volume_{key}"] = volume
        save_dict[f"ap_{key}"] = projections[0]
        save_dict[f"lat_{key}"] = projections[1]
    np.savez_compressed(filename, **save_dict)
    print(f"\u2705 Data saved to {filename}")

# ====== Run ======
if __name__ == "__main__":
    phantom_dataset = generate_dataset()
    save_to_npz(phantom_dataset, "phantom_data_testing.npz")
