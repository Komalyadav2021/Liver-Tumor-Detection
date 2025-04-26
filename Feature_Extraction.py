import os
import cv2
import torch
import pandas as pd
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Custom Dataset Class ---
class TumorDataset(Dataset):
    def __init__(self, raw_root, mask_root, transform):
        self.samples = []
        self.transform = transform

        for folder in os.listdir(raw_root):
            raw_path = os.path.join(raw_root, folder)
            mask_folder = folder.replace("volume", "segmentation")
            mask_path = os.path.join(mask_root, mask_folder)

            if not os.path.isdir(raw_path) or not os.path.isdir(mask_path):
                continue

            for file in os.listdir(raw_path):
                if not file.endswith('.png'):
                    continue

                raw_file = os.path.join(raw_path, file)
                mask_file = os.path.join(mask_path, file.replace("volume", "segmentation"))

                if os.path.exists(raw_file) and os.path.exists(mask_file):
                    self.samples.append((raw_file, mask_file, folder, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, mask_path, volume, filename = self.samples[idx]
        image = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel
        image = self.transform(image)
        tumor_present = int(np.any(mask > 0))
        
        mask_folder = os.path.basename(os.path.dirname(mask_path))  
        return image, tumor_present, volume, filename, mask_folder

# --- Custom collate_fn ---
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    images, labels, volumes, filenames, mask_folders = zip(*batch)
    return images, labels, volumes, filenames, mask_folders

# --- Load Pretrained ResNet50 ---
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
resnet.eval()

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet stds
])

# --- Paths ---
base_path = "paste your train dataset path"
raw_root = os.path.join(base_path, "raw_images")
mask_root = os.path.join(base_path, "mask")

dataset = TumorDataset(raw_root, mask_root, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

# --- Feature Extraction ---
features = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

with torch.no_grad():
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        imgs, labels, volumes, filenames, mask_folders = batch
        imgs = torch.stack(imgs).to(device)

        outputs = resnet(imgs).squeeze()  # shape: (batch, 2048)

        # Handle case when batch size is 1 (squeeze results in 1D tensor)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)

        for i in range(len(outputs)):
            feature_vec = outputs[i].cpu().numpy()
            features.append({
                'volume': volumes[i],
                'filename': filenames[i],
                'mask_folder': mask_folders[i],  # ✅ Corrected line
                'tumor_present': labels[i],
                **{f'feat_{j}': feature_vec[j] for j in range(len(feature_vec))}
            })

# --- Save to CSV ---
df = pd.DataFrame(features)
output_dir = os.path.join(base_path, "output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "deep_features_resnet.csv")
df.to_csv(output_path, index=False)

print(f"\n Deep feature extraction complete.\n Saved at: {output_path}")
