import os
import sys
from datetime import datetime
from typing import List, Any
import torch
from torch import nn, Tensor
from abc import abstractmethod
from torch import nn
from torch.nn import functional as F

from base import BaseModel
from logger import TensorboardWriter
from config.const import ROOT_DIR_PATH
import numpy as np

from utils import MetricTracker


class BaseVAE(BaseModel):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh())

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                in_channels=self.in_channels,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
            )
        )
        return data

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class LogCoshVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha: float = 100.,
                 beta: float = 10.,
                 **kwargs) -> None:
        super(LogCoshVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                in_channels=self.in_channels,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                alpha=self.alpha,
            )
        )
        return data

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        t = recons - input
        # recons_loss = F.mse_loss(recons, input)
        # cosh = torch.cosh(self.alpha * t)
        # recons_loss = (1./self.alpha * torch.log(cosh)).mean()

        recons_loss = self.alpha * t + \
                      torch.log(1. + torch.exp(- 2 * self.alpha * t)) - \
                      torch.log(torch.tensor(2.0))
        # print(self.alpha* t.max(), self.alpha*t.min())
        recons_loss = (1. / self.alpha) * recons_loss.mean()

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.beta * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.Capacity_max_iter = Capacity_max_iter
        self.loss_type = loss_type

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                in_channels=self.in_channels,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                beta=self.beta,
                gamma=self.gamma,
                max_capacity=self.max_capacity,
                Capacity_max_iter=self.Capacity_max_iter,
                loss_type=self.loss_type,
            )
        )
        return data

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def prepare_path(path, **args):
    res = os.path.join(ROOT_DIR_PATH, path.format(**args))
    create_dirs([res])
    return res


def get_env_parameters(env):
    # Get spaces parameters
    observation_dim = env.observation_space.spaces['observation'].shape[0]
    goal_dim = env.observation_space.spaces['desired_goal'].shape[0]

    action_dim = env.action_space.shape[0]
    max_action_value = float(env.action_space.high[0])

    return observation_dim, goal_dim, action_dim, max_action_value


class ReplayBuffer:
    def __init__(self, env, max_episode_steps, buffer_size, sample_func, device, image_shape):
        self.observation_dim, self.goal_dim, self.action_dim, self.action_max = get_env_parameters(env)
        self.device = device

        self.max_episode_steps = max_episode_steps
        self.size = buffer_size // self.max_episode_steps
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info

        self.observation_image_memory = np.empty([self.size, self.max_episode_steps, *image_shape], dtype=np.float32)
        self.observation_memory = np.empty([self.size, self.max_episode_steps, self.observation_dim], dtype=np.float32)
        self.achieved_goal_memory = np.empty([self.size, self.max_episode_steps, self.goal_dim], dtype=np.float32)
        self.desired_goal_memory = np.empty([self.size, self.max_episode_steps, self.goal_dim], dtype=np.float32)
        self.actions_memory = np.empty([self.size, self.max_episode_steps, self.action_dim], dtype=np.float32)

    # store the episode
    def store_episode(self, observation, image, achieved_goal, desired_goal, action, n_episodes_to_store):
        ids = self._get_storage_idx(inc=n_episodes_to_store)
        # store the information
        self.observation_image_memory[ids] = image
        self.observation_memory[ids] = observation
        self.achieved_goal_memory[ids] = achieved_goal
        self.desired_goal_memory[ids] = desired_goal
        self.actions_memory[ids] = action

        self.n_transitions_stored += self.max_episode_steps * n_episodes_to_store

    # sample the data from the replay buffer
    def sample(self, batch_size):
        observation_image_buffer = self.observation_image_memory[:self.current_size]
        observation_buffer = self.observation_memory[:self.current_size]
        achieved_goal_buffer = self.achieved_goal_memory[:self.current_size]
        desired_goal_buffer = self.desired_goal_memory[:self.current_size]
        actions_buffer = self.actions_memory[:self.current_size]

        return self.sample_func(observation_buffer,
                                observation_image_buffer,
                                achieved_goal_buffer, desired_goal_buffer,
                                actions_buffer,
                                batch_size)

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx


class SimpleSampler:
    def __init__(self, seq_len, reward_func):
        self.seq_len = seq_len
        self.reward_func = reward_func

    def sample_transitions(self, observation_buffer, observation_image_buffer,
                           achieved_goal_buffer, desired_goal_buffer,
                           actions_buffer, batch_size):
        # Trajectory length
        trajectory_length = actions_buffer.shape[1]

        # Buffer size
        buffer_length = actions_buffer.shape[0]

        # generate ids which trajectories to use
        episode_ids = np.random.randint(low=0, high=buffer_length, size=batch_size)

        # generate ids which timestamps to use
        # - 2 because we sample for 3 sequential timestamps
        t_samples = np.random.randint(low=0, high=trajectory_length - self.seq_len, size=batch_size)

        sequential_batches = []
        for time_i in range(self.seq_len):
            t_i = self._sample_for_time(observation_buffer, observation_image_buffer,
                                        achieved_goal_buffer, desired_goal_buffer, actions_buffer,
                                        episode_ids, t_samples,
                                        batch_size=batch_size, time=time_i)
            sequential_batches.append(t_i)

        (_, _, achieved_goal_batch_t1, desired_goal_batch_t1, _) = sequential_batches[-2]

        reward_batch = self.reward_func(achieved_goal_batch_t1, desired_goal_batch_t1, info=None)
        # Recompute the termination state for the augmented 'desired_goal'
        done_batch = reward_batch == 0

        # Reshape the batch
        reward_batch = reward_batch.reshape(batch_size, *reward_batch.shape[1:])
        done_batch = done_batch.reshape(batch_size, *done_batch.shape[1:])

        if len(done_batch.shape) == 1:
            done_batch = done_batch.reshape(batch_size, 1)

        if len(reward_batch.shape) == 1:
            reward_batch = reward_batch.reshape(batch_size, 1)

        return sequential_batches, reward_batch, done_batch

    def _sample_for_time(self, observation_buffer, observation_image_buffer,
                         achieved_goal_buffer, desired_goal_buffer, actions_buffer,
                         episode_idxs, t_samples, batch_size, time):
        observation_batch = observation_buffer[:, time:, :][episode_idxs, t_samples].copy()
        observation_image_batch = observation_image_buffer[:, time:, :][episode_idxs, t_samples].copy()
        achieved_goal_batch = achieved_goal_buffer[:, time:, :][episode_idxs, t_samples].copy()
        desired_goal_batch = desired_goal_buffer[:, time:, :][episode_idxs, t_samples].copy()
        actions_batch = actions_buffer[:, time:, :][episode_idxs, t_samples].copy()

        # Reshape the batch
        observation_batch = observation_batch.reshape(batch_size, *observation_batch.shape[1:])
        observation_image_batch = observation_image_batch.reshape(batch_size, *observation_image_batch.shape[1:])
        achieved_goal_batch = achieved_goal_batch.reshape(batch_size, *achieved_goal_batch.shape[1:])
        desired_goal_batch = desired_goal_batch.reshape(batch_size, *desired_goal_batch.shape[1:])
        actions_batch = actions_batch.reshape(batch_size, *actions_batch.shape[1:])

        return observation_batch, observation_image_batch, achieved_goal_batch, desired_goal_batch, actions_batch


class EpisodeData:

    def __init__(self):
        self.image_observation = []
        self.observation = []
        self.achieved_goal = []
        self.desired_goal = []
        self.action = []

    def add(self, observation, image_observation, achieved_goal, desired_goal, action):
        self.observation.append(observation)
        self.image_observation.append(image_observation)
        self.achieved_goal.append(achieved_goal)
        self.desired_goal.append(desired_goal)
        self.action.append(action)

    def as_numpy_arrays(self):
        return np.asarray(self.observation), np.asarray(self.image_observation), np.asarray(self.achieved_goal), \
               np.asarray(self.desired_goal), np.asarray(self.action)

    @classmethod
    def as_dict_of_numpy_arrays(cls, collected_step_episodes):
        data = EpisodeData()

        for elem in collected_step_episodes:
            data.add(*elem.as_numpy_arrays())

        data_as_dict = data.__dict__
        for key, value in data.__dict__.items():
            data_as_dict[key] = np.asarray(value)
        return data_as_dict


def interpret_boolean(param):
    if type(param) == bool:
        return param
    elif param in ['True', 'true', '1']:
        return True
    elif param in ['False', 'false', '0']:
        return False
    else:
        sys.exit("param '{}' cannot be interpreted as boolean".format(param))


def as_tensor(numpy_array, device):
    return torch.tensor(numpy_array, dtype=torch.float32, device=device)


class VAETrainer:
    def __init__(self, env, config):
        self.env = env
        self.vae_type = config.hparams.vae_type

        self.experiment_name = config.experiment_name
        self.vae_seq_len = config.vae.hparams.vae_seq_len  # The discount rate
        self._max_episode_steps = env._max_episode_steps

        self.n_epochs = int(config.hparams.n_epochs)
        self.steps_per_epoch = int(config.hparams.steps_per_epoch)
        self.n_rollout_episodes = int(config.hparams.n_rollout_episodes)
        self.n_training_iterations = int(config.hparams.n_training_iterations)

        self.should_save_model = interpret_boolean(config.should_save_model)
        self.model_path = prepare_path(config.model_path, experiment_name=config.experiment_name)
        self.model_save_timer = int(config.model_save_timer)

        self.batch_size = int(config.vae.hparams.batch_size)
        self.memory_capacity = int(config.vae.hparams.memory_capacity)
        self.image_shape = int(config.vae.hparams.image_shape)

        if self.vae_type == 'original':
            self.images_vae = VanillaVAE(3, 64)
        elif self.vae_type == 'LogCosh':
            self.images_vae = LogCoshVAE(3, 64)
        elif self.vae_type == 'LogCosh':
            self.images_vae = BetaVAE(3, 64)

        self.sampler = SimpleSampler(self.vae_seq_len,
                                     self.env.compute_reward)
        # create the replay buffer
        self.buffer = ReplayBuffer(self.env, self._max_episode_steps, self.memory_capacity,
                                   self.sampler.sample_transitions, config.device_id, self.image_shape)

        self.writer = TensorboardWriter(prepare_path(config.vae.tb_log_folder, experiment_name=config.experiment_name),
                                        True)

        self.train_metrics = MetricTracker('loss', 'val/loss', writer=self.writer)
        self.device = config.device_id

    def _reset(self):
        self.train_metrics.reset()
        native_observation = self.env.reset()

        observation = native_observation['observation']
        achieved_goal = native_observation['achieved_goal']
        desired_goal = native_observation['desired_goal']

        return observation, achieved_goal, desired_goal, False, 0

    def _preprocess_batch_images(self, sequence_of_batches):
        final_batch = []
        for time in range(len(sequence_of_batches)):
            (_, image_batch, _, _, _) = sequence_of_batches[time]
            final_batch.append(as_tensor(image_batch, self.device))
        return torch.stack(final_batch, dim=1)

    def _update_network(self):
        (sequential_batches, _, _) = self.buffer.sample(self.batch_size)

        encoder_input = self._preprocess_batch_images(sequential_batches) # (batch, seq, h, w, c)
        encoder_input = encoder_input.view(vae_batch_size, self.c, self.h, self.w, self.n_screens)

        recon, _, mu, logvar = self.images_vae.forward(obs_batch)
        loss = torch.mean(self.images_vae.loss_function(recon, obs_batch, mu, logvar))

        self.vae.optimizer.zero_grad()
        loss.backward()
        self.vae.optimizer.step()

        metrics = dict(
            loss=vfe.item()
        )
        return metrics

    def train(self):
        print("Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.now()))

        for epoch in range(self.n_epochs):
            for cycle in range(self.steps_per_epoch):
                step = self.steps_per_epoch * epoch + cycle
                self.writer.set_step(step)

                collected_step_episodes = []
                for _ in range(self.n_rollout_episodes):

                    episode_data = EpisodeData()
                    observation, achieved_goal, desired_goal, done, reward = self._reset()

                    for episode_step in range(self._max_episode_steps):
                        action = self.env.action_space.sample()

                        # feed the actions into the environment
                        new_observation, _, _, _ = self.env.step(action)

                        image_obs = self.env.render(mode='rgb_array')
                        image_obs = np.resize(np.cast(image_obs, ), self.image_shape)
                        image_obs /= 255
                        episode_data.add(observation.copy(), image_obs.copy(),
                                         achieved_goal.copy(), desired_goal.copy(), action.copy())

                        observation = new_observation['observation']
                        achieved_goal = new_observation['achieved_goal']

                    collected_step_episodes.append(episode_data)

                collected_step_episodes = EpisodeData.as_dict_of_numpy_arrays(collected_step_episodes)

                # store the episodes
                self.buffer.store_episode(**collected_step_episodes, n_episodes_to_store=self.n_rollout_episodes)

                train_iteration_metrics = []
                for _ in range(self.n_training_iterations):
                    # train the network
                    metrics_dict = self._update_network()
                    train_iteration_metrics.append(metrics_dict)

                train_iteration_metrics = {k: [dic[k] for dic in train_iteration_metrics]
                                           for k in train_iteration_metrics[0]}

                for metric, value in train_iteration_metrics.items():
                    self.train_metrics.update(metric, np.mean(value))

                print("Epoch: {:4d}, Step: {:4d}, metrics: {}".format(epoch, cycle, train_iteration_metrics))

            if self.should_save_model and epoch > 0 and epoch % self.model_save_timer == 0:
                epoch_path = self.model_path + "/epoch_" + str(epoch)
                create_dirs([epoch_path])
                self.images_vae.save(os.path.join(epoch_path, 'images_vae.pth'))

        self.env.close()
        print("Training finished at {}".format(datetime.now()))
