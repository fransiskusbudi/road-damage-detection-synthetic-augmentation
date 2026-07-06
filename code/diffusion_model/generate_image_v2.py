import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from PIL import Image
import os

# 1. Load the trained UNet model
model_path = "ddpm_checkpoints_v2/epoch_199"  # Change this to the correct model path
model = UNet2DModel.from_pretrained(model_path)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# 2. Create the pipeline
pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)

# 3. Move to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# 4. Generate synthetic pothole images
num_images = 542 # Set how many images you want
outdir = "fake_images_epoch_199"
os.makedirs(outdir, exist_ok=True)

for i in range(num_images):
    # Generate random noise as input (Ensuring correct dtype and device)
    latents = torch.randn((1, 3, 128, 128), dtype=torch.float32, device=device)

    # Generate the image
    output = pipe()  # Call the pipeline without arguments
    generated_image = output.images[0]  # Extract the first image

    # Convert the tensor output to a valid image format if necessary
    if isinstance(generated_image, torch.Tensor):
        generated_image = (generated_image * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        generated_image = Image.fromarray(generated_image)

    # Save the image
    generated_image.save(f"{outdir}/generated_{i}.png")

print("Images generated successfully!")
