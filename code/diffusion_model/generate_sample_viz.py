import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

# 1. Load the trained UNet model
model_path = "ddpm_checkpoints_1000/epoch_999"  # Change this to the correct model path
model = UNet2DModel.from_pretrained(model_path)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# 2. Create the pipeline
pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)

# 3. Move to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# 4. Set up directories for saving images
num_images = 1  # Only generate 1 image for visualization
outdir = "fake_images_progress"
os.makedirs(outdir, exist_ok=True)

# 5. Generate random noise
latents = torch.randn((1, 3, 128, 128), dtype=torch.float32, device=device)

# 6. Initialize a figure for visualization
num_steps_to_save = 10  # Number of intermediate steps to visualize
timesteps_to_save = torch.linspace(0, noise_scheduler.num_train_timesteps - 1, num_steps_to_save).long()

# 7. Denoising Process Visualization
for i in range(num_images):
    latents_copy = latents.clone()  # Make a copy to preserve original noise
    intermediate_images = []

    for t in reversed(range(noise_scheduler.num_train_timesteps)):  # Reverse for denoising
        # Predict noise
        with torch.no_grad():
            noise_pred = model(latents_copy, torch.tensor([t], device=device)).sample

        # Remove noise using DDPM scheduler
        latents_copy = noise_scheduler.step(noise_pred, t, latents_copy).prev_sample

        # Save intermediate images at selected timesteps
        if t in timesteps_to_save:
            img_np = latents_copy[0].permute(1, 2, 0).detach().cpu().numpy()
            img_np = np.clip((img_np - img_np.min()) / (img_np.max() - img_np.min()), 0, 1)  # Normalize to [0,1]

            intermediate_images.append(img_np)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(f"{outdir}/step_{t}.png")

print("Intermediate images saved successfully!")
