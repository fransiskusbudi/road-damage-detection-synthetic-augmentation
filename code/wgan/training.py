import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from tqdm import tqdm  # ✅ Add tqdm for progress tracking

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=40, critic_iterations=4, print_every=50,
                 use_cuda=False, multi_gpu=False):
        self.use_cuda = use_cuda
        self.multi_gpu = multi_gpu
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        # ✅ Multi-GPU Support
        if self.use_cuda and multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.G = nn.DataParallel(generator)
            self.D = nn.DataParallel(discriminator)
        else:
            self.G = generator
            self.D = discriminator

        self.G_opt = gen_optimizer
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        """ Train the Critic (Discriminator) """
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()

        d_real = self.D(data)
        d_generated = self.D(generated_data)

        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Compute loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        """ Train the Generator """
        self.G_opt.zero_grad()

        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        interpolated = (alpha * real_data.data + (1 - alpha) * generated_data.data).requires_grad_(True)

        prob_interpolated = self.D(interpolated)

        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones_like(prob_interpolated),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, epoch):
        """ Train for One Epoch """
        d_loss_total, g_loss_total, gp_total = 0.0, 0.0, 0.0
        batch_count = 0

        with tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch") as tepoch:
            for data in tepoch:
                batch_count += 1
                self.num_steps += 1

                self._critic_train_iteration(data)
                d_loss_total += self.losses['D'][-1]
                gp_total += self.losses['GP'][-1]

                if self.num_steps % self.critic_iterations == 0:
                    self._generator_train_iteration(data)
                    g_loss_total += self.losses['G'][-1]

                # Update progress bar
                tepoch.set_postfix(D_loss=d_loss_total / batch_count, G_loss=g_loss_total / batch_count)

        # Print summary at the end of the epoch
        avg_d_loss = d_loss_total / batch_count
        avg_g_loss = g_loss_total / batch_count
        avg_gp_loss = gp_total / batch_count

        print(f"\n[EPOCH {epoch+1}] Summary:")
        print(f"  - Avg D Loss: {avg_d_loss:.4f}")
        print(f"  - Avg G Loss: {avg_g_loss:.4f}")
        print(f"  - Avg GP Loss: {avg_gp_loss:.4f}")

    def train(self, data_loader, epochs, save_training_gif=True):
        """ Train the Model """
        if save_training_gif:
            fixed_latents = Variable(self.G.sample_latent(64) if not isinstance(self.G, torch.nn.DataParallel) else self.G.module.sample_latent(64))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
            self._train_epoch(data_loader, epoch)

            if save_training_gif and (epoch + 1) % 100 == 0:  # Save only every 10 epochs
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                training_progress_images.append(img_grid)

            if epoch + 1 == 2000:
                self.save_checkpoint(epoch + 1)

        if save_training_gif and training_progress_images:
            # Convert images from float32 (0-1) to uint8 (0-255)
            training_progress_images = [(img * 255).astype(np.uint8) for img in training_progress_images]

            # Save GIF
            imageio.mimsave(f'./training_{epochs}_epochs.gif', training_progress_images)


    def sample_generator(self, num_samples):
        """ Generate Fake Images """
        latent_samples = Variable(self.G.sample_latent(num_samples) if not isinstance(self.G, torch.nn.DataParallel) else self.G.module.sample_latent(num_samples))

        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        return self.G(latent_samples)

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        return generated_data.data.cpu().numpy()[:, 0, :, :]

    def save_checkpoint(self, epochs):
        """Save the current training state as a checkpoint."""
        name = 'pothole400'
        torch.save(self.G.state_dict(), './gen_' + name + '_epoch_' + str(epochs) + '.pt')
        torch.save(self.D.state_dict(), './dis_' + name + '_epoch_' + str(epochs) + '.pt')
        print(f"Checkpoint saved at epoch {epochs}")