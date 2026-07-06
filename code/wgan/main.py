# import torch
# import torch.optim as optim
# from dataloaders import get_mnist_dataloaders, get_lsun_dataloader, get_pothole_dataloader
# from models import Generator, Discriminator
# from training import Trainer

# data_loader = get_pothole_dataloader(batch_size=2)
# img_size = (128, 128, 3)

# generator = Generator(img_size=img_size, latent_dim=256, dim=64)
# discriminator = Discriminator(img_size=img_size, dim=64)

# print(generator)
# print(discriminator)

# # Initialize optimizers
# lr = 2e-4
# # betas = (.5, .9)
# # G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
# # D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
# D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.000005, betas=(0.5, 0.9))
# G_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))

# # Train model
# epochs = 3000
# trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
#                   use_cuda=True,multi_gpu=True)
# trainer.train(data_loader, epochs, save_training_gif=True)

# # Save models
# name = 'pothole'
# torch.save(trainer.G.state_dict(), './gen_' + name + str(epochs) + '.pt')
# torch.save(trainer.D.state_dict(), './dis_' + name + str(epochs) + '.pt')

import torch
import torch.optim as optim
from dataloaders import get_pothole_dataloader
from models import Generator, Discriminator
from training import Trainer

# Set parameters
batch_size = 16
img_size = (128, 128, 3)
latent_dim = 256
dim = 64
epochs = 4000
checkpoint_path_G = ""
checkpoint_path_D = ""
# checkpoint_path_G_new =

# Load data
data_loader = get_pothole_dataloader(batch_size=batch_size)

# Initialize Generator and Discriminator
generator = Generator(img_size=img_size, latent_dim=latent_dim, dim=dim)
discriminator = Discriminator(img_size=img_size, dim=dim)

# ✅ Load saved weights if available
try:
    generator.load_state_dict(torch.load(checkpoint_path_G))
    discriminator.load_state_dict(torch.load(checkpoint_path_D))
    print("✅ Successfully loaded pretrained Generator & Discriminator.")
except FileNotFoundError:
    print("❌ No pretrained models found. Starting training from scratch.")

# Initialize optimizers
D_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.9))
G_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))

# Resume training
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=True, multi_gpu=True)
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models again after training
name = 'pothole400'
torch.save(trainer.G.state_dict(), './gen_' + name + str(epochs) + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + str(epochs) + '.pt')

print(f"✅ Training completed! Models saved")
