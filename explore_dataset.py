import os
import numpy as np
from skimage.io import imread
import rasterio
import matplotlib.pyplot as plt

# Define dataset paths
base_path = r"C:\Users\LENOVO\Desktop\Land Usage AI\SEN-2 LULC"
folders = [
    ("train_images", ".png"),
    ("train_masks", ".tif"),
    ("val_images", ".png"),
    ("val_masks", ".tif"),
    ("test_images", ".png"),
    ("test_masks", ".tif")
]

# Count files and list samples
print("Number of files in each folder:")
for folder, ext in folders:
    folder_path = os.path.join(base_path, folder)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
    print(f"{folder}: {len(files)} files: {files[:5]}")

# Load a sample image and mask
train_image_path = os.path.join(base_path, "train_images")
train_mask_path = os.path.join(base_path, "train_masks")
image_files = [f for f in os.listdir(train_image_path) if f.lower().endswith(".png")]
mask_files = [f for f in os.listdir(train_mask_path) if f.lower().endswith(".tif")]

if not image_files or not mask_files:
    print("Error: No image or mask files found!")
    exit()

# Find a matching pair
for img_file in image_files:
    base_name = os.path.splitext(img_file)[0]
    mask_file = f"{base_name}.tif"
    if mask_file in mask_files:
        sample_image_path = os.path.join(train_image_path, img_file)
        sample_mask_path = os.path.join(train_mask_path, mask_file)
        break
else:
    print("Error: No matching image/mask pair found!")
    exit()

# Read image and mask
image = imread(sample_image_path)  # Shape: (height, width, channels)
with rasterio.open(sample_mask_path) as src:
    mask = src.read(1)  # Shape: (height, width)

# Print details
print("\nSample image shape:", image.shape)
print("Sample mask shape:", mask.shape)
print("Unique values in mask (land use classes):", np.unique(mask))

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Sample Image")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title("Sample Mask")
plt.imshow(mask, cmap="tab10")
plt.savefig("sample_image_mask.png")
plt.show()