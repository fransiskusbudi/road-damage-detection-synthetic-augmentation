import os

# Paths to your image and mask folders
image_folder = 'japan_diffusion_540_filtered_0.26/images'
mask_folder = 'japan_diffusion_540_filtered_0.26/labels'

# Sort files to maintain the pairing order
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

# Safety check: Ensure same number of images and masks
assert len(image_files) == len(mask_files), "Number of images and masks do not match!"

# Rename loop
for idx, (img_file, mask_file) in enumerate(zip(image_files, mask_files), start=1):
    # New filenames
    new_img_name = f"{idx}.jpg"
    new_mask_name = f"{idx}.txt"  # Change to .jpg if your masks are .jpg

    # Rename image
    os.rename(os.path.join(image_folder, img_file), os.path.join(image_folder, new_img_name))
    # Rename mask
    os.rename(os.path.join(mask_folder, mask_file), os.path.join(mask_folder, new_mask_name))

print("✅ Renaming completed! Files renamed from 1 to", idx)
