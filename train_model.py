import os
import numpy as np
from skimage.io import imread
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define paths
base_path = r"C:\Users\LENOVO\Desktop\Land Usage AI\SEN-2 LULC"
train_image_dir = os.path.join(base_path, "train_images")
train_mask_dir = os.path.join(base_path, "train_masks")

# Parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 64, 64, 3  # From Step 2
NUM_CLASSES = 7  # From Step 2: [1, 2, 3, 4, 5, 6, 7]
BATCH_SIZE = 4
EPOCHS = 5

# Data loading function
def load_data(image_dir, mask_dir, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images, masks = [], []
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.tif', '.tiff'))])
    
    print(f"Found {len(image_files)} image files: {image_files[:5]}")
    print(f"Found {len(mask_files)} mask files: {mask_files[:5]}")
    
    if not image_files or not mask_files:
        raise ValueError("No image or mask files found!")
    
    # Pair files numerically
    pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.tif"
        if mask_file in mask_files:
            pairs.append((img_file, mask_file))
    
    print(f"Found {len(pairs)} paired files: {pairs[:5]}")
    
    for img_file, mask_file in pairs[:50]:  # Limit to 50 for speed
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        print(f"Loading {img_file} and {mask_file}")
        
        # Load image
        try:
            img = imread(img_path)  # Shape: (height, width, channels)
            img = img / 255.0  # Normalize
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            continue
        # Load mask
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

# Build U-Net model
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

# Load data
print("Loading data...")
try:
    train_images, train_masks = load_data(train_image_dir, train_mask_dir)
    print("Train images shape:", train_images.shape)
    print("Train masks shape:", train_masks.shape)
except Exception as e:
    print(f"Data loading failed: {e}")
    exit()

# Build and compile model
model = build_unet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save('unet_model.h5')
model.save('unet_model.h5')

#added line

# Train model
print("Training model...")
history = model.fit(
    train_images, train_masks,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=1
)

# Save training plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig("training_plot.png")
plt.show()

# Predict on a sample
sample_image = train_images[0:1]
pred_mask = model.predict(sample_image)
pred_mask = np.argmax(pred_mask, axis=-1)[0]

# Save prediction plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Sample Image")
plt.imshow(sample_image[0])
plt.subplot(1, 3, 2)
plt.title("True Mask")
plt.imshow(np.argmax(train_masks[0], axis=-1), cmap="tab10")
plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="tab10")
plt.savefig("prediction_plot.png")
plt.show()