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

def main():
    # --- 1. Hyperparameters ---
    image_size = 128
    batch_size = 16
    num_epochs = 1000
    lr = 1e-4

    # --- 2. Create dataset & dataloader ---
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = PotholeDataset("cropped_50", transform=transform, max_images=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # --- 3. Define the model, noise scheduler, optimizer ---
    model = UNet2DModel(
        sample_size=image_size,  # the generated image resolution
        in_channels=3,           # 3 RGB channels
        out_channels=3,          # output channels = 3
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),  # increase or adjust as needed
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- 4. Prepare Accelerator for multi-GPU training if needed ---
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

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
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Optional: Save model checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"ddpm_checkpoints/epoch_{epoch}")

    print("Training complete!")

if __name__ == "__main__":
    main()
