import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class VAE(nn.Module):
    # In part taken from:
    #   https://github.com/pytorch/examples/blob/master/vae/main.py

    def __init__(self, n_screens, n_latent_states, lr=1e-5, device='cpu'):
        super(VAE, self).__init__()

        self.device = device

        self.n_screens = n_screens
        self.n_latent_states = n_latent_states

        # The convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, (5, 5, 1), (2, 2, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, (5, 5, 1), (2, 2, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, (5, 5, 1), (2, 2, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        ).to(self.device)

        # The size of the encoder output
        self.conv3d_shape_out = (32, 2, 8, self.n_screens)
        self.conv3d_size_out = np.prod(self.conv3d_shape_out)

        # The convolutional decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 32, (5, 5, 1), (2, 2, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, (5, 5, 1), (2, 2, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 3, (5, 5, 1), (2, 2, 1)),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),

            nn.Sigmoid()
        ).to(self.device)

        # Fully connected layers connected to encoder
        self.fc1 = nn.Linear(self.conv3d_size_out, self.conv3d_size_out // 2)
        self.fc2_mu = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)
        self.fc2_logvar = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)

        # Fully connected layers connected to decoder
        self.fc3 = nn.Linear(self.n_latent_states, self.conv3d_size_out // 2)
        self.fc4 = nn.Linear(self.conv3d_size_out // 2, self.conv3d_size_out)

        self.optimizer = optim.Adam(self.parameters(), lr)

        self.to(self.device)

    def encode(self, x):
        # Deconstruct input x into a distribution over latent states
        conv = self.encoder(x)
        h1 = F.relu(self.fc1(conv.view(conv.size(0), -1)))
        mu, logvar = self.fc2_mu(h1), self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Apply reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, batch_size=1):
        # Reconstruct original input x from the (reparameterized) latent states
        h3 = F.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view([batch_size] + [dim for dim in self.conv3d_shape_out])
        y = self.decoder(deconv_input)
        return y

    def forward(self, x, batch_size=1):
        # Deconstruct and then reconstruct input x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, batch_size)
        return recon, mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, batch=True):
        if batch:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
            BCE = torch.sum(BCE, dim=(1, 2, 3, 4))

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        else:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
