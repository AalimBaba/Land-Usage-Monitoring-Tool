import os
import numpy as np
from skimage.io import imread
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define paths
base_path = r"C:\Users\LENOVO\Desktop\Land Usage AI\SEN-2 LULC"
test_image_dir = os.path.join(base_path, "test_images")
test_mask_dir = os.path.join(base_path, "test_masks")
model_path = r"C:\Users\LENOVO\Desktop\Land Usage AI\unet_model.h5"

# Parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 64, 64, 3
NUM_CLASSES = 7
BATCH_SIZE = 4

# Data loading function
def load_data(image_dir, mask_dir, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images, masks = [], []
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.tif', '.tiff'))])
    
    print(f"Found {len(image_files)} image files: {image_files[:5]}")
    print(f"Found {len(mask_files)} mask files: {mask_files[:5]}")
    
    pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.tif"
        if mask_file in mask_files:
            pairs.append((img_file, mask_file))
    
    print(f"Found {len(pairs)} paired files: {pairs[:5]}")
    
    for img_file, mask_file in pairs[:20]:  # Limit to 20 for evaluation
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        print(f"Loading {img_file} and {mask_file}")
        
        try:
            img = imread(img_path)  # Shape: (height, width, channels)
            img = img / 255.0
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            continue
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Shape: (height, width)
                mask = mask - 1  # Shift [1, 2, ..., 7] to [0, 1, ..., 6]
        except Exception as e:
            print(f"Error loading mask {mask_file}: {e}")
            continue
        
        images.append(img)
        masks.append(mask)
    
    if not images:
        raise ValueError("No valid image/mask pairs loaded!")
    
    images = np.array(images)
    masks = np.array(masks)
    print("Unique mask values:", np.unique(masks))
    masks = tf.keras.utils.to_categorical(masks, num_classes=NUM_CLASSES)
    return images, masks

# Build U-Net model (same as train_model.py)
def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    u4 = layers.UpSampling2D(2)(c3)
    u4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)
    u5 = layers.UpSampling2D(2)(c4)
    u5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(c5)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c5)
    
    model = models.Model(inputs, outputs)
    return model

# Save the trained model (modify train_model.py to save)
print("Modifying train_model.py to save the model...")
with open(r"C:\Users\LENOVO\Desktop\Land Usage AI\train_model.py", 'r') as file:
    lines = file.readlines()
with open(r"C:\Users\LENOVO\Desktop\Land Usage AI\train_model.py", 'w') as file:
    for line in lines:
        file.write(line)
        if "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])" in line:
            file.write("    model.save('unet_model.h5')\n")

# Load data
print("Loading test data...")
try:
    test_images, test_masks = load_data(test_image_dir, test_mask_dir)
    print("Test images shape:", test_images.shape)
    print("Test masks shape:", test_masks.shape)
except Exception as e:
    print(f"Data loading failed: {e}")
    exit()

# Load model
print("Loading model...")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Model loading failed: {e}. Rebuilding and using untrained model.")
    model = build_unet()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate model
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(test_images, test_masks, batch_size=BATCH_SIZE, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on samples
print("Generating predictions...")
pred_masks = model.predict(test_images[:5])  # Predict on 5 samples
pred_masks = np.argmax(pred_masks, axis=-1)
true_masks = np.argmax(test_masks[:5], axis=-1)

# Save prediction plots
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(5, 3, i*3 + 1)
    plt.title("Test Image")
    plt.imshow(test_images[i])
    plt.axis('off')
    plt.subplot(5, 3, i*3 + 2)
    plt.title("True Mask")
    plt.imshow(true_masks[i], cmap="tab10")
    plt.axis('off')
    plt.subplot(5, 3, i*3 + 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_masks[i], cmap="tab10")
    plt.axis('off')
plt.tight_layout()
plt.savefig("test_predictions.png")
plt.show()