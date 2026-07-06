import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import os

# Import your Generator model (Ensure models.py is in the same directory)
from models import Generator  

# === PARAMETERS ===
IMG_SIZE = (128, 128, 3)  # Change based on your training setup
LATENT_DIM = 256          # Change based on your training setup
NUM_IMAGES = 1000          # Number of images to generate
MODEL_PATH = "gen_pothole400_epoch_4000.pt"  # Path to trained generator
FAKE_IMAGE_FOLDER = "generated_fake_images1000"  # Folder to save generated images
OUTPUT_COLLAGE = "collated_image400_2000_fid.png"  # Final output image

# === 1️⃣ LOAD TRAINED GENERATOR ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(img_size=IMG_SIZE, latent_dim=LATENT_DIM, dim=64)  # Adjust 'dim' if needed

# Load trained weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
generator.load_state_dict(checkpoint)
generator.eval().to(device)  # Set to evaluation mode and move to device

# === 2️⃣ GENERATE IMAGES ===
# Create output folder if not exists
os.makedirs(FAKE_IMAGE_FOLDER, exist_ok=True)

# Generate latent vectors
latent_vectors = torch.randn(NUM_IMAGES, LATENT_DIM).to(device)

# Generate images
with torch.no_grad():
    fake_images = generator(latent_vectors)

# Convert images from [-1,1] to [0,255] and save them
fake_images = fake_images.permute(0, 2, 3, 1).cpu().numpy()  # Convert (B, C, H, W) -> (B, H, W, C)
fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)  # Normalize and convert to uint8

for i in range(NUM_IMAGES):
    imageio.imwrite(os.path.join(FAKE_IMAGE_FOLDER, f"generated_{i}.png"), fake_images[i])

print(f"{NUM_IMAGES} images generated and saved in '{FAKE_IMAGE_FOLDER}/'.")

# # === 3️⃣ COLLATE IMAGES INTO ONE GRID ===
# GRID_ROWS = 10  # Number of rows in the grid
# GRID_COLS = 10  # Number of columns in the grid

# # Get list of saved images
# image_files = sorted([os.path.join(FAKE_IMAGE_FOLDER, f) for f in os.listdir(FAKE_IMAGE_FOLDER) if f.endswith(".png")])[:NUM_IMAGES]

# # Load images
# images = [imageio.imread(img) for img in image_files]

# # Get image dimensions
# img_height, img_width, _ = images[0].shape  # Assuming all images are the same size

# # Create blank canvas for the collage
# collated_image = np.zeros((GRID_ROWS * img_height, GRID_COLS * img_width, 3), dtype=np.uint8)

# # Place images into the grid
# for idx, img in enumerate(images):
#     row = idx // GRID_COLS
#     col = idx % GRID_COLS
#     collated_image[row * img_height: (row + 1) * img_height, col * img_width: (col + 1) * img_width, :] = img

# # Save the final collage image
# imageio.imwrite(OUTPUT_COLLAGE, collated_image)

# # Display the final collage
# plt.figure(figsize=(10, 10))
# plt.imshow(collated_image)
# plt.axis("off")
# plt.show()

# print(f"Collated image saved as '{OUTPUT_COLLAGE}'.")
