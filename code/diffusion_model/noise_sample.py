

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Path to the image
image_path = "/home/USER/diffusion/cropped_400/Japan_000723_crop0.jpg"  # Replace with actual image path
output_noise_path = "/home/USER/diffusion/noise_image.jpg"
output_noisy_image_path = "/home/USER/diffusion/noisy_image.jpg"

# Load the image
image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format

# Transform to tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to (C, H, W) tensor in [0,1]
])

image_tensor = transform(image).unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
batch_size, channels, height, width = image_tensor.shape

# Generate noise
noise = torch.randn(image_tensor.shape)  # Gaussian noise with same shape

# Normalize noise to [0, 1] for visualization
noise_min = noise.min()
noise_max = noise.max()
noise_normalized = (noise - noise_min) / (noise_max - noise_min)  # Normalize to [0,1]

# Add noise to the image and clamp values to [0,1]
noisy_image = image_tensor + noise * 0.2  # Adjust noise intensity (0.2 factor)
noisy_image = torch.clamp(noisy_image, 0, 1)  # Ensure values stay in [0,1]

# Convert to NumPy for saving
noise_np = (noise_normalized[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
noisy_image_np = (noisy_image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

# Convert NumPy arrays to PIL Images
noise_pil = Image.fromarray(noise_np)
noisy_image_pil = Image.fromarray(noisy_image_np)

# Save images
noise_pil.save(output_noise_path)
noisy_image_pil.save(output_noisy_image_path)

print(f"Noise image saved at: {output_noise_path}")
print(f"Noisy image saved at: {output_noisy_image_path}")
