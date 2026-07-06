import torch
import open_clip
import os
from PIL import Image
from tqdm import tqdm
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
clip_model_path = '/home/USER/mlpractical_rdd/pothole_augmentation/pretrained_models/clip_models/ViT-B-32.pt'
image_folder = 'outputs/bdd_diffusion_540_filtered_0.26'
# output_folder = 'wgan_filtered'
# os.makedirs(output_folder, exist_ok=True)

# Load OpenCLIP model with local weights
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=clip_model_path)
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Prepare text embedding for "real pothole"
text = ["A real road pothole"]
text_tokens = tokenizer(text).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# prompts = [
#     "A large pothole on an asphalt road",
#     "A damaged road with a pothole",
#     "A deep pothole in the middle of the street",
#     "Cracks and potholes on the road surface",
#     "A road with visible pothole damage",
#     "A close-up of a road pothole",
#     "Severe road damage caused by potholes",
#     "An old road with cracks and potholes",
#     "A pothole surrounded by asphalt cracks"
# ]
# text_tokens = tokenizer(prompts).to(device)
# with torch.no_grad():
#     text_features = model.encode_text(text_tokens)  # shape: [num_prompts, 512]
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#     avg_text_feature = text_features.mean(dim=0, keepdim=True)  # shape: [1, 512]
#     avg_text_feature = avg_text_feature / avg_text_feature.norm(dim=-1, keepdim=True)
#     text_features = avg_text_feature


# List all image files in the synthetic folder
synthetic_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]

image_scores = []

for img_path in tqdm(synthetic_images):
    try:
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity with the text embedding
            similarity = (image_features @ text_features.T).squeeze().item()
            image_scores.append((img_path, similarity))
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

# Sort images by similarity (optional)
image_scores.sort(key=lambda x: x[1], reverse=True)

# Filter and save images with similarity above threshold
threshold = 0.26
filtered_count = 0

for img_path, score in image_scores:
    if score > threshold:
        # Copy the image to the filtered folder
        # shutil.copy(img_path, os.path.join(output_folder, os.path.basename(img_path)))
        filtered_count += 1

        print(f"{os.path.basename(img_path)}: Similarity Score = {score:.4f}")

print(f"\nTotal images filtered (score > {threshold}): {filtered_count}")
