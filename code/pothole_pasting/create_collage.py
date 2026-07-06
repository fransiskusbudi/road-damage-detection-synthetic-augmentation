import os
import random
from PIL import Image
import math

# Folders
folders = {
    'potholes/diffusion_filtered_0.26_540': 'diffusion_collage.png',
    'potholes/wgan_filtered_generated_images_540_0.28': 'wgan_collage.png'
}

# Parameters
samples_per_collage = 100
thumb_size = (128, 128)

for folder, output_file in folders.items():
    # Collect image paths
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Safety check
    if len(images) < samples_per_collage:
        raise ValueError(f"Not enough images in {folder}. Found {len(images)} but need {samples_per_collage}.")

    # Random sample
    sampled_images = random.sample(images, samples_per_collage)

    # Grid size (e.g., 10x10 for 100 images)
    grid_size = int(math.ceil(math.sqrt(samples_per_collage)))
    collage_width = thumb_size[0] * grid_size
    collage_height = thumb_size[1] * grid_size

    # Create blank canvas
    collage = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    # Paste images into the grid
    for idx, img_path in enumerate(sampled_images):
        img = Image.open(img_path).convert("RGB")
        img = img.resize(thumb_size)

        x = (idx % grid_size) * thumb_size[0]
        y = (idx // grid_size) * thumb_size[1]
        collage.paste(img, (x, y))

    # Save the collage
    collage.save(output_file)
    print(f"✅ Collage saved as {output_file}")
