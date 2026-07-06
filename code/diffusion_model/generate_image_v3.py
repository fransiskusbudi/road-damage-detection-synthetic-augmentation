# import torch
# from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
# import open_clip
# from PIL import Image
# from tqdm import tqdm
# import os
# import shutil

# # === Setup ===
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # DDPM Model
# ddpm_model_path = "ddpm_checkpoints_1000/epoch_999"
# unet_model = UNet2DModel.from_pretrained(ddpm_model_path)
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# pipe = DDPMPipeline(unet=unet_model, scheduler=noise_scheduler).to(device)

# # CLIP Model
# clip_model_path = '/home/USER/mlpractical_rdd/pothole_augmentation/pretrained_models/clip_models/ViT-B-32.pt'
# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=clip_model_path)
# clip_model = clip_model.to(device)
# clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# # CLIP Text Embedding
# text = ["A real road pothole"]
# text_tokens = clip_tokenizer(text).to(device)
# with torch.no_grad():
#     text_features = clip_model.encode_text(text_tokens)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

# # Output folders
# save_dir = "filtered_generated_images"
# os.makedirs(save_dir, exist_ok=True)

# # === Target Setup ===
# target_images = 500
# clip_threshold = 0.26  # Adjust if needed
# filtered_count = 0
# attempt_count = 0

# print("Starting generation and filtering...")

# with tqdm(total=target_images, desc="Filtered Images Collected") as pbar:
#     while filtered_count < target_images:
#         attempt_count += 1
#         # Generate image with DDPM
#         output = pipe()
#         generated_image = output.images[0]

#         # Convert to PIL Image if tensor
#         if isinstance(generated_image, torch.Tensor):
#             generated_image = (generated_image * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
#             generated_image = Image.fromarray(generated_image)

#         # CLIP Preprocessing
#         image_input = clip_preprocess(generated_image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             image_features = clip_model.encode_image(image_input)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             similarity = (image_features @ text_features.T).squeeze().item()

#         if similarity > clip_threshold:
#             # Save the filtered image
#             generated_image.save(os.path.join(save_dir, f"filtered_generated_{filtered_count}.png"))
#             filtered_count += 1
#             pbar.update(1)
#             print(f"✅ Saved Image {filtered_count} (Attempt {attempt_count}): Similarity = {similarity:.4f}")
#         else:
#             print(f"❌ Discarded (Attempt {attempt_count}): Similarity = {similarity:.4f}")

# print(f"\n✅ Generation complete! Total attempts: {attempt_count}")
# print(f"✅ {filtered_count} images saved in '{save_dir}' (CLIP score > {clip_threshold})")
# # 

import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import open_clip
from PIL import Image
from tqdm import tqdm
import os

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# DDPM Model
ddpm_model_path = "ddpm_checkpoints_1000/epoch_999"
unet_model = UNet2DModel.from_pretrained(ddpm_model_path)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
pipe = DDPMPipeline(unet=unet_model, scheduler=noise_scheduler).to(device)

# CLIP Model
clip_model_path = '/home/USER/mlpractical_rdd/pothole_augmentation/pretrained_models/clip_models/ViT-B-32.pt'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=clip_model_path)
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# CLIP Text Embedding
text = ["A real road pothole"]
text_tokens = clip_tokenizer(text).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Output folder
save_dir = "filtered_generated_images"
os.makedirs(save_dir, exist_ok=True)

# === Target Setup ===
target_images = 540
clip_threshold = 0.26  # Adjust if needed
attempt_count = 0

print(f"Starting generation with dynamic filename checking AFTER scoring (parallel-safe)...")

with tqdm(desc="Filtered Images Collected") as pbar:
    while True:
        # Dynamically check how many images exist before generation
        existing_images = sorted([f for f in os.listdir(save_dir) if f.endswith(".png")])
        filtered_count = len(existing_images)
        
        if filtered_count >= target_images:
            break  # Stop once total images reach target

        attempt_count += 1

        # === 1. Generate image ===
        output = pipe()
        generated_image = output.images[0]

        # Convert to PIL Image if tensor
        if isinstance(generated_image, torch.Tensor):
            generated_image = (generated_image * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            generated_image = Image.fromarray(generated_image)

        # === 2. CLIP Scoring ===
        image_input = clip_preprocess(generated_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze().item()

        if similarity > clip_threshold:
            # === 3. ONLY NOW check the latest filename before saving ===
            current_files = sorted([f for f in os.listdir(save_dir) if f.endswith(".png")])
            if current_files:
                latest_index = max([int(f.split('_')[-1].split('.')[0]) for f in current_files])
                next_index = latest_index + 1
            else:
                next_index = 0

            # Save the filtered image with the correct dynamic index
            generated_image.save(os.path.join(save_dir, f"filtered_generated_{next_index}.png"))
            pbar.update(1)
            print(f"✅ Saved Image {next_index} (Attempt {attempt_count}): Similarity = {similarity:.4f}")
        else:
            print(f"❌ Discarded (Attempt {attempt_count}): Similarity = {similarity:.4f}")

print(f"\n✅ Generation complete! Total attempts: {attempt_count}")
print(f"✅ Total images in '{save_dir}': {target_images} (CLIP score > {clip_threshold})")
