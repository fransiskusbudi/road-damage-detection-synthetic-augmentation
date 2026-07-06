import torch
import numpy as np
import os
import imageio.v2 as imageio
from PIL import Image
import open_clip
from tqdm import tqdm

# === PARAMETERS ===
IMG_SIZE = (128, 128, 3)
LATENT_DIM = 256
TARGET_IMAGES = 40  # Number of images you want to save after filtering
MODEL_PATH = "gen_pothole400_epoch_4000.pt"
SAVE_DIR = "wgan_filtered_generated_images_1000_threshold_0.30"
CLIP_THRESHOLD = 0.28  # Adjust the similarity threshold

# === LOAD GENERATOR ===
from models import Generator  # Ensure your Generator model is correct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(img_size=IMG_SIZE, latent_dim=LATENT_DIM, dim=64).to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
generator.eval()

# === SETUP CLIP ===
clip_model_path = '/home/USER/mlpractical_rdd/pothole_augmentation/pretrained_models/clip_models/ViT-B-32.pt'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=clip_model_path)
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Prepare text embedding for scoring
text = ["A real road pothole"]
text_tokens = clip_tokenizer(text).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# === Create output folder ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === GENERATION AND FILTERING LOOP ===
filtered_count = 0
attempt_count = 0

print("Starting WGAN generation and CLIP filtering...")

with tqdm(total=TARGET_IMAGES, desc="Filtered Images Collected") as pbar:
    while filtered_count < TARGET_IMAGES:
        attempt_count += 1
        # Generate a latent vector and a fake image
        z = torch.randn(1, LATENT_DIM).to(device)
        with torch.no_grad():
            fake_image = generator(z)  # Output shape: (1, C, H, W)
            fake_image = fake_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            fake_image = ((fake_image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)  # [-1,1] -> [0,255]
            image_pil = Image.fromarray(fake_image)

        # CLIP Preprocessing and scoring
        image_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze().item()

        if similarity > CLIP_THRESHOLD:
            # Save the filtered image
            save_path = os.path.join(SAVE_DIR, f"filtered_generated_{filtered_count+500}.png")
            image_pil.save(save_path)
            filtered_count += 1
            pbar.update(1)
            print(f"✅ Saved Image {filtered_count} (Attempt {attempt_count}): Similarity = {similarity:.4f}")
        else:
            print(f"❌ Discarded (Attempt {attempt_count}): Similarity = {similarity:.4f}")

print(f"\n✅ WGAN Generation complete! Total attempts: {attempt_count}")
print(f"✅ {filtered_count} images saved in '{SAVE_DIR}' (CLIP score > {CLIP_THRESHOLD})")
