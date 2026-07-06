# import os
# import random
# import shutil

# # Paths
# dataset_folder = 'train_no_pothole/images'
# output_folder = 'train_no_pothole_250'
# num_samples = 100  # Change this to the number of images you want to pick

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Get list of all image files in the dataset folder
# image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# # Check if the dataset has enough images
# if num_samples > len(image_files):
#     raise ValueError(f"Requested {num_samples} samples, but only {len(image_files)} images are available.")

# # Randomly select images
# selected_images = random.sample(image_files, num_samples)

# # Copy selected images to the output folder
# for img_file in selected_images:
#     src_path = os.path.join(dataset_folder, img_file)
#     dst_path = os.path.join(output_folder, img_file)
#     shutil.copy(src_path, dst_path)

# print(f"✅ {num_samples} random images copied to '{output_folder}/'")

import os
import random
import shutil

# Paths
image_folder = 'dataset_poisson_blend/images'
mask_folder = 'dataset_poisson_blend/mask'
output_image_folder = 'dataset_japan_no_pothole_540/images'
output_mask_folder = 'dataset_japan_no_pothole_540/mask'
num_samples = 540  # Set how many pairs you want to pick

# Create output directories
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Check dataset size
if num_samples > len(image_files):
    raise ValueError(f"Requested {num_samples} samples, but only {len(image_files)} images are available.")

# Randomly select image files
selected_files = random.sample(image_files, num_samples)

# Copy image-mask pairs
for img_filename in selected_files:
    # Compute mask filename
    mask_filename = f"da_{img_filename}"
    
    # Verify the mask exists
    mask_path = os.path.join(mask_folder, mask_filename)
    if not os.path.exists(mask_path):
        print(f"⚠️ Mask not found for {img_filename}, skipping...")
        continue

    # Copy image
    shutil.copy(os.path.join(image_folder, img_filename),
                os.path.join(output_image_folder, img_filename))
    # Copy corresponding mask
    shutil.copy(mask_path, os.path.join(output_mask_folder, mask_filename))

print(f"✅ {num_samples} random image-mask pairs (with 'da_' masks) copied to 'random_selected_dataset/'")