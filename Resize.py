# import os
# import cv2

# # Paths to original directories (containing subfolders for each patient)
# image_root_dir = r"E:/Liver_Tumor_Research/3Dircadb1/3Dircadb1/3Dircadb1.3/PATIENT_PNG"
# mask_root_dir = r"E:/Liver_Tumor_Research/3Dircadb1/3Dircadb1/3Dircadb1.3/MASKS_PNG"

# # Output root directories for resized images and masks
# output_image_128 = r"E:/Liver_Tumor_Research/Resized_Images_128"
# output_image_256 = r"E:/Liver_Tumor_Research/Resized_Images_256"

# output_mask_128 = r"E:/Liver_Tumor_Research/Resized_Masks_128"
# output_mask_256 = r"E:/Liver_Tumor_Research/Resized_Masks_256"

# # Ensure output root directories exist
# os.makedirs(output_image_128, exist_ok=True)
# os.makedirs(output_image_256, exist_ok=True)
# os.makedirs(output_mask_128, exist_ok=True)
# os.makedirs(output_mask_256, exist_ok=True)

# # Function to recursively resize images in all subdirectories
# def resize_images_recursive(input_root, output_root_128, output_root_256):
#     for root, dirs, files in os.walk(input_root):  # Recursively walk through directories
#         for filename in files:
#             if filename.endswith(".png"):
#                 img_path = os.path.join(root, filename)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#                 if img is None:
#                     print(f"Skipping {filename}, unable to load image.")
#                     continue

#                 # Resize images
#                 img_128 = cv2.resize(img, (128, 128))
#                 img_256 = cv2.resize(img, (256, 256))

#                 # Preserve subfolder structure
#                 relative_path = os.path.relpath(root, input_root)  # Get relative path
#                 output_subfolder_128 = os.path.join(output_root_128, relative_path)
#                 output_subfolder_256 = os.path.join(output_root_256, relative_path)

#                 os.makedirs(output_subfolder_128, exist_ok=True)
#                 os.makedirs(output_subfolder_256, exist_ok=True)

#                 # Save resized images
#                 cv2.imwrite(os.path.join(output_subfolder_128, filename), img_128)
#                 cv2.imwrite(os.path.join(output_subfolder_256, filename), img_256)

#         print(f"Processed: {root}")

# # Resize both images and masks for all patients (including subfolders)
# resize_images_recursive(image_root_dir, output_image_128, output_image_256)
# resize_images_recursive(mask_root_dir, output_mask_128, output_mask_256)

# print("Resizing completed for all patient data.")


import os
import cv2

# Root directory to search for MASKS_PNG
root_dir = r"E:/Liver_Tumor_Research/3Dircadb1/3Dircadb1/"

# Output directories for resized masks
output_mask_128 = r"E:/Liver_Tumor_Research/Resized_Masks1_128"
output_mask_256 = r"E:/Liver_Tumor_Research/Resized_Masks1_256"

# Ensure output directories exist
os.makedirs(output_mask_128, exist_ok=True)
os.makedirs(output_mask_256, exist_ok=True)

# Function to resize images
def resize_images(input_root, output_root_128, output_root_256):
    for root, _, files in os.walk(input_root):  # Recursively walk through directories
        for filename in files:
            if filename.endswith(".png"):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Skipping {filename}, unable to load image.")
                    continue

                # Resize images
                img_128 = cv2.resize(img, (128, 128))
                img_256 = cv2.resize(img, (256, 256))

                # Preserve subfolder structure inside single output folder
                relative_path = os.path.relpath(root, input_root).replace("\\", "_").replace("/", "_")
                new_filename_128 = f"{relative_path}_{filename}"
                new_filename_256 = f"{relative_path}_{filename}"

                # Save resized images
                cv2.imwrite(os.path.join(output_root_128, new_filename_128), img_128)
                cv2.imwrite(os.path.join(output_root_256, new_filename_256), img_256)

        print(f"Processed: {input_root}")

# Detect all MASKS_PNG folders and resize their images
for root, dirs, _ in os.walk(root_dir):
    if "MASKS_PNG" in dirs:
        masks_path = os.path.join(root, "MASKS_PNG")
        print(f"Found MASKS_PNG at: {masks_path}")
        resize_images(masks_path, output_mask_128, output_mask_256)

print("Resizing completed for all detected MASKS_PNG folders.")
