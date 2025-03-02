import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Skipping {image_path}, unable to load image.")
        return None

    # Feature 1: Mean Intensity
    mean_intensity = np.mean(image)

    # Feature 2: Histogram (Flattened)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # Feature 3: Canny Edge Detection
    edges = cv2.Canny(image, 50, 150)
    edge_count = np.sum(edges > 0)

    # Feature 4: Texture Features (GLCM - Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    return {
        "Image Name": os.path.basename(image_path),
        "Mean Intensity": mean_intensity,
        "Edge Count": edge_count,
        "Contrast": contrast,
        "Energy": energy
    }

# **Process All Images in a Folder**
def process_images(folder_path):
    features_list = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)
            if features:
                features_list.append(features)
    
    return features_list

# **Set Your Folder Path Containing PNG Images**
folder_path = r"E:\Liver_Tumor_Research\Liver and Tumor Segmentation_All_Code\Converted_Images"
all_features = process_images(folder_path)

# **Convert Features to a Pandas DataFrame**
df = pd.DataFrame(all_features)

# **Save the DataFrame to an Excel File**
output_path = r"E:\Liver_Tumor_Research\Liver and Tumor Segmentation_All_Code\Extracted_Features.xlsx"
df.to_excel(output_path, index=False)

print(f"Feature extraction complete. Data saved to {output_path}")
