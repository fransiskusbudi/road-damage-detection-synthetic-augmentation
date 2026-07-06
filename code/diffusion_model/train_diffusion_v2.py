import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch_fidelity import calculate_metrics
import torch_fidelity

class PotholeDataset(Dataset):
    def __init__(self, folder, transform=None, max_images=None):
        super().__init__()
        self.folder = folder
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        # Apply limit only if max_images is specified and greater than 0
        if max_images and max_images > 0:
            self.files = self.files[:max_images]

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def generate_fid_samples(model, noise_scheduler, epoch, num_samples, accelerator, image_size=128):
    # Unwrap the model from DDP (required for proper execution in single-GPU inference)
    model = accelerator.unwrap_model(model)

    # Initialize the diffusion pipeline
    pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipe.to(accelerator.device)  # Move to the correct device

    # Output directory
    outdir = f"samples/epoch_{epoch}"
    os.makedirs(outdir, exist_ok=True)

    print(f"Generating {num_samples} images for FID evaluation (Epoch {epoch})...")

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Generating FID Samples (Epoch {epoch})"):
            # Generate the image using the diffusion pipeline (no `latents` argument)
            output = pipe()  # This will internally sample noise and generate an image
            generated_image = output.images[0]  # Extract the first image

            # Convert tensor to PIL image (Ensure it is in range [0, 255])
            if isinstance(generated_image, torch.Tensor):
                generated_image = (generated_image * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                generated_image = Image.fromarray(generated_image)

            # Save the image
            generated_image.save(f"{outdir}/generated_{i}.png")

    print(f"Generated {num_samples} images in {outdir}")

def main():
    # --- 1. Hyperparameters ---
    image_size = 128
    batch_size = 16
    num_epochs = 1000
    lr = 1e-4

    # --- 2. Create dataset & dataloader ---
    transform = transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = PotholeDataset("cropped_400", transform=transform, max_images=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Total images: {len(dataset)}")

    # --- 3. Define the model, noise scheduler, optimizer ---
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_epochs * len(dataloader),
    )

    # --- 4. Prepare Accelerator for multi-GPU training if needed ---
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    

    # --- 5. Training Loop ---
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # 5.1. Sample noise
            noise = torch.randn(batch.shape).to(batch.device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=batch.device).long()

            # 5.2. Add noise to images
            noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

            # 5.3. Predict the noise using the model
            model_output = model(noisy_images, timesteps).sample

            # 5.4. Loss: mean squared error between the predicted noise and actual noise
            loss = torch.nn.functional.mse_loss(model_output, noise)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1

        # Optional: Save model checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"ddpm_checkpoints_1000/epoch_{epoch}")

        if epoch >= 99 and (epoch + 1) % 200  == 0:  # Compute FID every 50 epochs starting from epoch 50
            model.eval()

            # Generate 10 images for FID (adjustable)
            num_fid_samples = 25
            generate_fid_samples(model, noise_scheduler, epoch, num_fid_samples, accelerator, image_size)

            metrics_dict = torch_fidelity.calculate_metrics(
                input1=f"samples/epoch_{epoch}",
                input2="cropped_400",  
                isc=True, 
                fid=True
            )


            print(f"Epoch {epoch} - {metrics_dict}")

            # Save FID score
            with open("fid_scores_1000_cropped_400.txt", "a") as f:
                f.write(f"Epoch {epoch}: {metrics_dict}\n")
    print("Training complete!")

if __name__ == "__main__":
    main()
