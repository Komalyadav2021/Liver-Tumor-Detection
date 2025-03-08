import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ============ CONFIGURATION =============
IMAGE_SIZE = 128  # Resize images to (128x128)
DATASET_PATH = r"E:\Liver_Tumor_Research\Resized_Masks1_128"
OUTPUT_CSV_PATH = r"E:\Liver_Tumor_Research\Feature_Extraction_Output1.csv"
BATCH_SIZE = 16

# ========== BUILD RESUNET FEATURE EXTRACTOR ==========
def residual_block(x, filters):
    """ A residual block with a skip connection to match shapes correctly. """
    skip = Conv2D(filters, (1, 1), padding="same")(x)  # Match shape

    x = Conv2D(filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)

    x = Add()([x, skip])  # Ensure both tensors have the same shape
    x = Activation("relu")(x)
    return x

def build_resunet_feature_extractor(input_shape):
    """ Builds a feature extractor using ResUNet. """
    inputs = Input(input_shape)

    x1 = residual_block(inputs, 64)
    x2 = residual_block(x1, 128)
    x3 = residual_block(x2, 256)

    x4 = residual_block(x3, 512)  # Final feature representation

    # Global Average Pooling to obtain meaningful feature representation
    pooled_features = GlobalAveragePooling2D()(x4)

    model = Model(inputs, pooled_features, name="ResUNet_FeatureExtractor")
    return model

# ========== LOAD & PREPROCESS DATA ==========
def load_images_from_folder(folder):
    """ Load all images from a folder, resize and convert to array. """
    images = []
    image_filenames = []

    for root, dirs, files in os.walk(folder):
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode="grayscale")  
                img_array = img_to_array(img) / 255.0  # Normalize
                images.append(img_array)
                image_filenames.append(filename)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    return np.array(images), image_filenames

print("Loading dataset...")
image_data, image_filenames = load_images_from_folder(DATASET_PATH)

# Ensure proper shape for grayscale images
image_data = np.expand_dims(image_data, axis=-1)  # Convert (N, 128, 128) → (N, 128, 128, 1)

print(f"Dataset loaded with {len(image_data)} images.")

# ========== FEATURE EXTRACTION ==========
print("Building ResUNet model for feature extraction...")
feature_extractor = build_resunet_feature_extractor((IMAGE_SIZE, IMAGE_SIZE, 1))
feature_extractor.summary()

print("Extracting features...")
features = feature_extractor.predict(image_data, batch_size=BATCH_SIZE)

# Convert to Pandas DataFrame
df = pd.DataFrame(features)
df.insert(0, "Filename", image_filenames)  # Add filenames as the first column

# Save to CSV
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Feature extraction completed. Output saved at: {OUTPUT_CSV_PATH}")
