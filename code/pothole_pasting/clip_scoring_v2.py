import torch
import open_clip
import os
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
clip_model_path = '/home/USER/mlpractical_rdd/pothole_augmentation/pretrained_models/clip_models/ViT-B-32.pt'
image_folder = 'potholes/wgan_filtered_generated_images_540_0.28'

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

# List all image files in the synthetic folder
synthetic_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]

image_scores = []
similarity_sum = 0
total_images = 0

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
            similarity_sum += similarity
            total_images += 1

            print(f"{os.path.basename(img_path)}: Similarity Score = {similarity:.4f}")

    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

# Calculate and print average similarity
avg_similarity = similarity_sum / total_images if total_images > 0 else 0
print(f"\n✅ Average CLIP Similarity Score across all images: {avg_similarity:.4f}")
