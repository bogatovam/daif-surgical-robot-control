import copy
import os.path
from abc import abstractmethod, ABC
from typing import Tuple, Optional, NamedTuple, List, Any

from stable_baselines3.common.utils import polyak_update

import surrol.gym as surrol_gym
from omegaconf import OmegaConf
from stable_baselines3.common.distributions import StateDependentNoiseDistribution, \
    TanhBijector, DiagGaussianDistribution, SquashedDiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose
from stable_baselines3.common.torch_layers import create_mlp

from base import BaseModel
from config.configs_reader import get_config

from config.const import ROOT_DIR_PATH
from logger import TensorboardWriter
from utils import MetricTracker, create_dirs

from datetime import datetime
import os.path

import gym

import os.path
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

import threading
import numpy as np
import imageio
import gym
import torch
from torch import nn, Tensor, optim
from abc import abstractmethod
from torch import nn
from torch.nn import functional as F


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
    def loss_function(self, *inputs: Any, **kwargs):
        pass


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 device='cpu',
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.device = device
        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=(3, 3, 1), stride=(2, 2, 1),
                              padding=(1, 1, 0)),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 16, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=(3, 3, 1),
                                       stride=(2, 2, 1),
                                       padding=(1, 1, 0),
                                       output_padding=(1, 1, 0), ),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=(3, 3, 1),
                               stride=(2, 2, 1),
                               padding=(1, 1, 0),
                               output_padding=(1, 1, 0)),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh())

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        print(self)
        self.to(self.device)

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
        result = result.view(-1, 256, 2, 2, 4)
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
                      **kwargs):
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
        return loss, recons_loss.detach(), -kld_loss.detach()

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
                 device='cpu',
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()
        self.device = device
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
            hidden_dims = [16, 32, 64, 128, 256]

            # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=(3, 3, 1), stride=(2, 2, 1),
                              padding=(1, 1, 0)),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 16, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=(3, 3, 1),
                                       stride=(2, 2, 1),
                                       padding=(1, 1, 0),
                                       output_padding=(1, 1, 0), ),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=(3, 3, 1),
                               stride=(2, 2, 1),
                               padding=(1, 1, 0),
                               output_padding=(1, 1, 0)),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh())
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.to(self.device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                devicie=self.device,
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
        result = result.view(-1, 256, 2, 2, 4)
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
                      **kwargs):
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
        return loss, recons_loss.detach(), -kld_loss.detach()

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
    def store_episode(self, observation, image_observation, achieved_goal, desired_goal, action, n_episodes_to_store):
        ids = self._get_storage_idx(inc=n_episodes_to_store)
        # store the information
        self.observation_image_memory[ids] = image_observation
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


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        # calculate the new mean and std
        self.mean = self.local_sum / self.local_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.local_sumsq / self.local_count) - np.square(
            self.local_sum / self.local_count)))

    # normalize the observation
    def normalize(self, v):
        return (v - self.mean) / np.clip(self.std, 1e-9, None)


class BasePolicy(BaseModel, ABC):

    def __init__(self,
                 observation_space,
                 action_space,
                 normalize_images: bool = True,
                 squash_output: bool = False):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images
        self._squash_output = squash_output

    @property
    def squash_output(self) -> bool:
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        raise NotImplementedError

    def predict(self, observation):
        return self._predict(observation)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def obs_to_tensor(self, observation):
        if isinstance(observation, dict):
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        return observation


class Actor(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            input_size,
            net_arch,
            action_distribution_type,
            device,
            weight_decay=0.00001,
            lr=0.0001,
            activation_fn=nn.ReLU,
            log_std_init: float = -1,
            full_std: bool = True,
            sde_net_arch=None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = True,
    ):
        super().__init__(observation_space, action_space, normalize_images=normalize_images, squash_output=True)

        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.action_distribution_type = action_distribution_type

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        action_dim = get_action_dim(self.action_space)
        # here will be vae
        self.latent_pi = nn.Sequential(*create_mlp(input_size, -1, net_arch, activation_fn))
        self.device = device
        last_layer_dim = net_arch[-1]

        if self.action_distribution_type == 'StateDependentNoiseDistribution':
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        elif self.action_distribution_type == 'SquashedDiagGaussianDistribution':
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_size = input_size
        self.net_arch = net_arch
        self.weight_decay = weight_decay
        self.lr = lr
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.clip_mean = clip_mean
        self.normalize_images = normalize_images

        self.to(device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                action_distribution_type=self.action_distribution_type,
                observation_space=self.observation_space,
                action_space=self.action_space,
                input_size=self.input_size,
                net_arch=self.net_arch,
                weight_decay=self.weight_decay,
                lr=self.lr,
                activation_fn=self.activation_fn,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                sde_net_arch=self.sde_net_arch,
                use_expln=self.use_expln,
                clip_mean=self.clip_mean,
                normalize_images=self.normalize_images,
                device=self.device,
            )
        )
        return data

    def get_std(self) -> torch.Tensor:
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, observations: torch.Tensor):
        # features = self.extract_features(observations)
        latent_pi = self.latent_pi(observations)
        mean_actions = self.mu(latent_pi)

        if self.action_distribution_type == 'StateDependentNoiseDistribution':
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)

        log_std = self.log_std(latent_pi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self(observation, deterministic)


class MLP(BaseModel):
    def __init__(self, input_size, layer_sizes, output_size, lr=0.0001, output_activation=torch.nn.Identity,
                 activation=torch.nn.ReLU, weight_decay=1e-6, device='cpu'):
        super(MLP, self).__init__()

        self.layers = create_mlp(input_size, output_size, layer_sizes, activation)
        self.layers = nn.ModuleList(self.layers)
        self.optimizer = optim.Adam(self.parameters(), lr)  # Adam optimizer

        self.device = device
        self.to(self.device)

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        self.lr = lr
        self.output_activation = output_activation
        self.activation = activation
        self.weight_decay = weight_decay

    def forward(self, inp):
        x = inp
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionModelMlpPreprocessor:
    def __init__(self, preprocess_func, device):
        super(TransitionModelMlpPreprocessor, self).__init__()
        self.preprocess_func = preprocess_func
        self.device = device

    def preprocess(self, sequence_of_batches):
        (observation_batch, _, desired_goal_batch, actions_batch_t0, _) = sequence_of_batches[0]
        state_batch_t0 = self.preprocess_func(observation_batch, desired_goal_batch)
        return torch.cat((state_batch_t0, as_tensor(actions_batch_t0, self.device)), dim=1)


class TransitionModelRnnPreprocessor:
    def __init__(self, preprocess_func, device):
        super(TransitionModelRnnPreprocessor, self).__init__()
        self.preprocess_func = preprocess_func
        self.device = device

    def preprocess(self, sequence_of_batches):
        final_batch = []
        for time in range(len(sequence_of_batches)):
            (observation_batch, _, desired_goal_batch, actions_batch, _) = sequence_of_batches[time]
            state_batch = self.preprocess_func(observation_batch, desired_goal_batch)
            final_batch.append(torch.cat((state_batch, as_tensor(actions_batch, self.device)), dim=1))
        return torch.stack(final_batch, dim=1)


class LSTM(BaseModel):
    def __init__(self, input_size, layer_sizes, output_size, lr=0.0001, output_activation=torch.nn.Identity,
                 activation=torch.nn.ReLU, drop_prob=0.2, device='cpu'):
        super(LSTM, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_sizes = layer_sizes
        self.layer_dim = len(layer_sizes)
        self.hidden_dim = layer_sizes[0]

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, self.hidden_dim, self.layer_dim, batch_first=True, dropout=drop_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr)  # Adam optimizer

        self.device = device
        self.to(self.device)

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        self.lr = lr
        self.drop_prob = drop_prob
        self.output_activation = output_activation
        self.activation = activation

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                input_size=self.input_size,
                layer_sizes=self.layer_sizes,
                output_size=self.output_size,
                lr=self.lr,
                output_activation=self.output_activation,
                activation=self.activation,
                drop_prob=self.drop_prob,
                device=self.device,
            )
        )
        return data


class GRU(BaseModel):
    def __init__(self, input_size, layer_sizes, output_size,
                 lr=0.0001, output_activation=torch.nn.Identity,
                 activation=torch.nn.ReLU, drop_prob=0.2, device='cpu'):
        super(GRU, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_sizes = layer_sizes
        self.layer_dim = len(layer_sizes)
        self.hidden_dim = layer_sizes[0]

        # GRU layers
        self.gru = nn.GRU(
            input_size, self.hidden_dim, self.layer_dim, batch_first=True, dropout=drop_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr)

        self.device = device
        self.to(self.device)

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        self.drop_prob = drop_prob
        self.lr = lr
        self.output_activation = output_activation
        self.activation = activation

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                input_size=self.input_size,
                layer_sizes=self.layer_sizes,
                output_size=self.output_size,
                lr=self.lr,
                output_activation=self.output_activation,
                activation=self.activation,
                drop_prob=self.drop_prob,
                device=self.device,
            )
        )
        return data


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


class EpisodeSummary:
    def __init__(self):
        self.done = []
        self.reward = []
        self.goal_distance = []
        self.jaw_state = []

    def add(self, reward, info):
        if 'goal_distance' in info:
            self.goal_distance.append(info['goal_distance'])
        if 'jaw_state' in info:
            self.jaw_state.append(info['jaw_state'])
        self.done.append(info['is_success'])
        self.reward.append(reward)

    def add_with_no_transformation(self, reward, done, goal_distance, jaw_state):
        self.reward.append(reward)
        self.done.append(done)
        self.goal_distance.append(goal_distance)
        self.jaw_state.append(jaw_state)

    def calc_summary(self):
        return np.mean(self.reward), np.mean(self.done), \
               np.mean(self.goal_distance) if len(self.goal_distance) > 0 else -1, \
               np.mean(self.jaw_state) if len(self.jaw_state) > 0 else -1

    @classmethod
    def as_dict_of_values(cls, collected_step_summary):
        data = EpisodeSummary()

        for elem in collected_step_summary:
            data.add_with_no_transformation(*elem.calc_summary())

        data_as_dict = data.__dict__
        for key, value in data.__dict__.items():
            data_as_dict[key] = np.mean(value)
        return data_as_dict


class SacMinimise:
    def __init__(self, actor, value_net, target_net, beta, gamma):
        self.actor = actor
        self.value_net = value_net
        self.target_net = target_net
        self.beta = beta
        self.gamma = gamma

    def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
                               actions_batch_t1,
                               reward_batch, done_batch,
                               pred_error_batch_t0t1, alpha):
        with torch.no_grad():
            actions_t2, log_prob_t2 = self.actor.action_log_prob(state_batch_t2)

            targe_net_input = torch.cat([state_batch_t2, actions_t2], dim=1)
            target_expected_free_energies_batch_t2 = self.target_net(targe_net_input)

            # H_t2 ~ -log_prob_t2
            weighted_targets = target_expected_free_energies_batch_t2 + alpha * log_prob_t2.reshape(-1, 1)

            # Determine the batch of bootstrapped estimates of the EFEs:
            expected_free_energy_estimate_batch = (
                    -reward_batch + pred_error_batch_t0t1 + (1 - done_batch) * self.beta * weighted_targets)

        # Determine the Expected free energy at time t1 according to the value network:
        value_net_input_t1 = torch.cat([state_batch_t1, actions_batch_t1], dim=1)
        value_net_output_t1 = self.value_net(value_net_input_t1)

        # Determine the MSE loss between the EFE estimates and the value network output:
        mse = 0.5 * F.mse_loss(expected_free_energy_estimate_batch, value_net_output_t1)
        return mse

    def compute_variational_free_energy(self, state_batch_t1, predicted_actions_t1, pred_log_prob_t1,
                                        pred_error_batch_t0t1, alpha):
        value_net_input = torch.cat([state_batch_t1, predicted_actions_t1], dim=1)
        expected_free_energy_t1 = self.value_net(value_net_input)

        vfe_batch = pred_error_batch_t0t1 + alpha * pred_log_prob_t1 + self.gamma * expected_free_energy_t1
        return torch.mean(vfe_batch)


class AdaptedDaif:

    def __init__(self, actor, value_net, target_net, beta, gamma):
        self.actor = actor
        self.value_net = value_net
        self.target_net = target_net
        self.beta = beta
        self.gamma = gamma

    def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
                               actions_batch_t1,
                               reward_batch, done_batch,
                               pred_error_batch_t0t1, alpha):
        with torch.no_grad():
            actions_t2, log_prob_t2 = self.actor.action_log_prob(state_batch_t2)

            targe_net_input = torch.cat([state_batch_t2, actions_t2], dim=1)
            target_expected_free_energies_batch_t2 = self.target_net(targe_net_input)

            weighted_targets = -log_prob_t2 * target_expected_free_energies_batch_t2

            expected_free_energy_estimate_batch = (
                    -reward_batch + pred_error_batch_t0t1 + (1 - done_batch) * self.beta * weighted_targets)

        value_net_input_t1 = torch.cat([state_batch_t1, actions_batch_t1], dim=1)
        value_net_output_t1 = self.value_net(value_net_input_t1)

        mse = 0.5 * F.mse_loss(expected_free_energy_estimate_batch, value_net_output_t1)
        return mse

    def compute_variational_free_energy(self, state_batch_t1, predicted_actions_t1, pred_log_prob_t1,
                                        pred_error_batch_t0t1, alpha):
        value_net_input = torch.cat([state_batch_t1, predicted_actions_t1], dim=1)
        expected_free_energy_t1 = self.value_net(value_net_input)

        # Weigh them according to the action distribution:
        energy_batch = (-self.gamma * expected_free_energy_t1)

        # Determine the entropy of the action distribution
        entropy_batch = -pred_log_prob_t1 * alpha

        # Determine the Variable Free Energy, then take the mean over all batch samples:
        vfe_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
        vfe = torch.mean(vfe_batch)
        return vfe


class Agent:
    def __init__(self, env, config):
        self.env = env
        self.experiment_name = config.experiment_name
        self.experiment_description = config.experiment_description
        self.observation_dim, self.goal_dim, self.action_dim, self.action_max = get_env_parameters(env)

        self.device = config.device_id

        self.polyak = int(config.hparams.polyak)

        self.actor_action_distribution = config.hparams.actor_action_distribution
        self.n_warmap_episodes = int(config.hparams.n_warmap_episodes)

        self.n_epochs = int(config.hparams.n_epochs)
        self.steps_per_epoch = int(config.hparams.steps_per_epoch)
        self.n_rollout_episodes = int(config.hparams.n_rollout_episodes)
        self.n_training_iterations = int(config.hparams.n_training_iterations)
        self._max_episode_steps = env._max_episode_steps

        self.batch_size = int(config.hparams.batch_size)
        self.memory_capacity = int(config.hparams.memory_capacity)

        self.gamma = float(config.hparams.gamma)  # A precision parameter
        self.beta = float(config.hparams.beta)  # The discount rate
        self.alpha = config.hparams.alpha  # The discount rate

        self.should_save_model = interpret_boolean(config.should_save_model)
        self.model_path = prepare_path(config.model_path, experiment_name=config.experiment_name)
        self.video_log_path = os.path.join(
            prepare_path(config.video_log_folder, experiment_name=config.experiment_name), "epoch-{}.gif")
        self.model_save_timer = int(config.model_save_timer)

        self.should_save_episode_video = interpret_boolean(config.should_save_episode_video)
        self.episode_video_timer = int(config.episode_video_timer)

        self.state_shape = np.add(self.env.observation_space['observation'].shape,
                                  self.env.observation_space['desired_goal'].shape)
        self.state_size = np.prod(self.state_shape)

        self.actions_shape = self.env.action_space.shape
        self.action_dim = self.env.action_space.shape[-1]
        self.observations_seq_len = config.hparams.observations_seq_len  # The discount rate

        assert (self.n_rollout_episodes >= self.observations_seq_len)
        assert (self.n_warmap_episodes >= self.observations_seq_len)

        self.current_epoch = 0
        self.actor = Actor(env.observation_space, env.action_space,
                           self.state_size,
                           OmegaConf.to_object(config.hparams.actor_layers),
                           action_distribution_type=self.actor_action_distribution,
                           lr=config.hparams.actor_lr,
                           device=self.device)

        self.value_net = MLP(self.state_size + self.action_dim,
                             OmegaConf.to_object(config.hparams.value_net_layers),
                             1,
                             lr=config.hparams.value_net_lr,
                             device=self.device)
        self.target_net = MLP(self.state_size + self.action_dim,
                              OmegaConf.to_object(config.hparams.value_net_layers),
                              1,
                              lr=config.hparams.value_net_lr,
                              device=self.device)

        self.transition_network_type = config.hparams.transition_network_type

        if self.transition_network_type == 'mlp':
            self.transition_net = MLP(self.state_size + self.action_dim,
                                      OmegaConf.to_object(config.hparams.transition_net_layers),
                                      self.state_size,
                                      lr=config.hparams.value_net_lr,
                                      device=self.device)
            self.transition_preprocessor = TransitionModelMlpPreprocessor(self._preprocess_batch_inputs, self.device)
        elif self.transition_network_type == 'lstm':
            self.transition_net = LSTM(self.state_size + self.action_dim,
                                       OmegaConf.to_object(config.hparams.transition_net_layers),
                                       self.state_size,
                                       lr=config.hparams.value_net_lr,
                                       device=self.device)
            self.transition_preprocessor = TransitionModelRnnPreprocessor(self._preprocess_batch_inputs, self.device)

        elif self.transition_network_type == 'gru':
            self.transition_net = GRU(self.state_size + self.action_dim,
                                      OmegaConf.to_object(config.hparams.transition_net_layers),
                                      self.state_size,
                                      lr=config.hparams.value_net_lr,
                                      device=self.device)
            self.transition_preprocessor = TransitionModelRnnPreprocessor(self._preprocess_batch_inputs, self.device)

        # entropy coeff settings
        self.log_alpha = None
        self.alpha_optimizer = None
        self.alpha_tensor = None
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        if isinstance(self.alpha, str) and self.alpha.startswith("auto"):
            init_value = 1.0
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.hparams.alpha_lr)
        else:
            self.alpha_tensor = torch.tensor(float(self.alpha)).to(self.device)

        if config.hparams.efe_approximation_approach == 'sac_maximize':
            self.efe_approximation_approach = SacMaximise(self.actor, self.value_net, self.target_net,
                                                          self.beta, self.gamma)
        elif config.hparams.efe_approximation_approach == 'sac_minimise':
            self.efe_approximation_approach = SacMinimise(self.actor, self.value_net, self.target_net,
                                                          self.beta, self.gamma)
        elif config.hparams.efe_approximation_approach == 'sac_minimize_entropy':
            self.efe_approximation_approach = SacMinimiseEntropy(self.actor, self.value_net, self.target_net,
                                                                 self.beta, self.gamma)
        elif config.hparams.efe_approximation_approach == 'adapted_daif':
            self.efe_approximation_approach = AdaptedDaif(self.actor, self.value_net, self.target_net,
                                                          self.beta, self.gamma)

        self.her_module = HERSampler(config.hparams.replay_strategy, config.hparams.replay_k, self.observations_seq_len,
                                     self.env.compute_reward)
        # create the replay buffer
        self.buffer = ReplayBuffer(self.env, self._max_episode_steps, self.memory_capacity,
                                   self.her_module.sample_her_transitions, config.device_id)

        self.o_norm = Normalizer(size=env.observation_space.spaces['observation'].shape[0])
        self.g_norm = Normalizer(size=env.observation_space.spaces['desired_goal'].shape[0])
        self.a_norm = Normalizer(size=self.action_dim)
        self.target_update_interval = 1

        self.writer = TensorboardWriter(prepare_path(config.tb_log_folder, experiment_name=config.experiment_name),
                                        True)

        self.train_metrics = MetricTracker('vfe', 'value_net_loss', 'alpha', 'alpha_loss', 'success_rate', 'reward',
                                           'transition_net_grad', 'actor_grad_acc', 'value_net_grad',
                                           'sde_std', 'transition_net_loss', 'goal_distance', 'jaw_state',
                                           writer=self.writer)

        self.val_metrics = MetricTracker('val/success_rate', 'val/reward', writer=self.writer)

        # just to save model configuration to logs
        self.config_as_dict = OmegaConf.to_object(config.hparams)
        self.config_as_dict['actor_layers'] = str(self.config_as_dict['actor_layers'])
        self.config_as_dict['transition_net_layers'] = str(self.config_as_dict['transition_net_layers'])
        self.config_as_dict['value_net_layers'] = str(self.config_as_dict['value_net_layers'])
        self.config_as_dict['max_episode_steps'] = str(env._max_episode_steps)

        with open(os.path.join(self.model_path, "config.yaml"), 'w+') as file:
            OmegaConf.save(config, file)

    def restore(self):
        self.transition_net = self.transition_net.load(os.path.join(self.model_path, 'transition_net.pth'), self.device)
        self.actor = self.actor.load(os.path.join(self.model_path, 'actor.pth'), self.device)
        self.value_net = self.value_net.load(os.path.join(self.model_path, 'value_net.pth'), self.device)
        self.target_net.load_state_dict(self.value_net.state_dict(), self.device)
        self.current_epoch = int(self.model_path.split('epoch-')[1]) + 1

        if self.log_alpha is not None:
            saved_log_alpha = torch.load(os.path.join(self.model_path, 'log_alpha.pt'))
            self.log_alpha = saved_log_alpha

        if self.alpha_optimizer is not None:
            saved_alpha_optimizer = torch.load(os.path.join(self.model_path, 'alpha_optimizer.pt'))
            self.alpha_optimizer.load_state_dict(state_dict=saved_alpha_optimizer["state_dict"])

        if self.alpha_tensor is not None:
            saved_alpha_tensor = torch.load(os.path.join(self.model_path, 'alpha_tensor.pt'))
            self.alpha_tensor = saved_alpha_tensor

    def get_mini_batches(self):
        (sequential_batches, reward_batch, done_batch) = self.buffer.sample(self.batch_size)

        transition_model_raw_input, t1, t2 = sequential_batches[:-2], sequential_batches[-2], sequential_batches[-1]

        (observation_batch_t1, achieved_goal_batch_t1, desired_goal_batch_t1, actions_batch_t1, _) = t1
        (observation_batch_t2, achieved_goal_batch_t2, desired_goal_batch_t2, actions_batch_t2, _) = t2

        state_batch_t1 = self._preprocess_batch_inputs(observation_batch_t1, desired_goal_batch_t1)
        state_batch_t2 = self._preprocess_batch_inputs(observation_batch_t2, desired_goal_batch_t2)

        transition_net_input = self.transition_preprocessor.preprocess(transition_model_raw_input)
        pred_batch_t0t1 = self.transition_net(transition_net_input)

        pred_error_batch_t0t1 = torch.mean(
            F.mse_loss(pred_batch_t0t1, state_batch_t1, reduction='none'), dim=1).unsqueeze(1)

        return (state_batch_t1, state_batch_t2,
                as_tensor(actions_batch_t1, self.device),
                as_tensor(reward_batch, self.device),
                as_tensor(done_batch, self.device),
                pred_error_batch_t0t1)

    def _update_network(self):
        if self.actor_action_distribution == 'StateDependentNoiseDistribution':
            self.actor.reset_noise()

        # Retrieve transition data in mini batches:
        (state_batch_t1, state_batch_t2, actions_batch_t1,
         reward_batch, done_batch, pred_error_batch_t0t1) = self.get_mini_batches()
        # Compute the value network loss:

        # Action by the current actor for the sampled state
        sampled_actions_t1, sampled_actions_log_prob_t1 = self.actor.action_log_prob(state_batch_t1)
        sampled_actions_log_prob_t1 = sampled_actions_log_prob_t1.reshape(-1, 1)

        alpha_loss = None
        if self.alpha_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha = torch.exp(self.log_alpha.detach())
            alpha_loss = -(self.log_alpha * (sampled_actions_log_prob_t1 + self.target_entropy).detach()).mean()
        else:
            alpha = self.alpha_tensor

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        value_net_loss = self.efe_approximation_approach.compute_value_net_loss(state_batch_t1, state_batch_t2,
                                                                                actions_batch_t1,
                                                                                reward_batch, done_batch,
                                                                                pred_error_batch_t0t1,
                                                                                alpha)
        self.transition_net.optimizer.zero_grad()

        self.value_net.optimizer.zero_grad()
        value_net_loss.backward()
        value_net_grad = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 100000.)
        self.value_net.optimizer.step()

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        # Compute the variational free energy:
        vfe = self.efe_approximation_approach.compute_variational_free_energy(state_batch_t1,
                                                                              sampled_actions_t1,
                                                                              sampled_actions_log_prob_t1,
                                                                              pred_error_batch_t0t1, alpha)
        vfe.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100000.)
        transition_net_grad = torch.nn.utils.clip_grad_norm_(self.transition_net.parameters(), 100000.)
        self.actor.optimizer.step()
        self.transition_net.optimizer.step()

        metrics = dict(
            vfe=vfe.item(),
            actor_grad_acc=actor_grad.item(),
            alpha=alpha.item(),
            alpha_loss=alpha_loss.detach().item(),
            value_net_loss=value_net_loss.item(),
            value_net_grad=value_net_grad.item(),
            transition_net_loss=pred_error_batch_t0t1.mean().item(),
            transition_net_grad=transition_net_grad.item(),
        )
        return metrics

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

    # do the evaluation
    def _eval_agent(self, epoch):
        images = []
        reward_array = []
        done = []
        episode_step = 0

        observation, _, desired_goal, _, _ = self._reset()
        while episode_step < self._max_episode_steps:
            input_tensor = self._preprocess_inputs(observation, desired_goal)
            action = self._select_action(input_tensor)

            new_observation, reward, _, info = self.env.step(action)

            observation = new_observation['observation']
            reward_array.append(reward)
            done.append(info['is_success'])
            episode_step += 1

            if self.should_save_episode_video and epoch % self.episode_video_timer == 0:
                images += [self.env.render(mode='rgb_array')]

        return np.mean(np.asarray(done)), np.mean(np.asarray(reward_array)), np.asarray(images)

    def train(self):
        self.writer.add_text(self.experiment_name, self.experiment_description)
        print("Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.now()))

        self.warmup()
        for epoch in range(self.current_epoch, self.n_epochs + 1):
            for cycle in range(self.steps_per_epoch):
                step = self.steps_per_epoch * epoch + cycle
                self.writer.set_step(step)

                collected_step_summary = []
                collected_step_episodes = []
                for _ in range(self.n_rollout_episodes):

                    episode_data, episode_summary = EpisodeData(), EpisodeSummary()
                    observation, achieved_goal, desired_goal, done, reward = self._reset()

                    for episode_step in range(self._max_episode_steps):
                        input_tensor = self._preprocess_inputs(observation, desired_goal)
                        action = self._select_action(input_tensor)

                        # feed the actions into the environment
                        new_observation, reward, _, info = self.env.step(action)

                        episode_data.add(observation.copy(), achieved_goal.copy(), desired_goal.copy(), action.copy(),
                                         info)
                        episode_summary.add(np.mean(reward), info)

                        observation = new_observation['observation']
                        achieved_goal = new_observation['achieved_goal']

                    collected_step_episodes.append(episode_data)
                    collected_step_summary.append(episode_summary)

                collected_step_episodes = EpisodeData.as_dict_of_numpy_arrays(collected_step_episodes)
                collected_step_summary = EpisodeSummary.as_dict_of_values(collected_step_summary)

                # store the episodes
                self.buffer.store_episode(**collected_step_episodes, n_episodes_to_store=self.n_rollout_episodes)
                self._update_normalizer(**collected_step_episodes)

                train_iteration_metrics = []
                for _ in range(self.n_training_iterations):
                    # train the network
                    metrics_dict = self._update_network()
                    train_iteration_metrics.append(metrics_dict)

                train_iteration_metrics = {k: [dic[k] for dic in train_iteration_metrics]
                                           for k in train_iteration_metrics[0]}

                for metric, value in train_iteration_metrics.items():
                    self.train_metrics.update(metric, np.mean(value))

                if self.actor_action_distribution == 'StateDependentNoiseDistribution':
                    self.train_metrics.update('sde_std', (self.actor.get_std()).mean().item())

                # soft update
                if cycle % self.target_update_interval == 0:
                    polyak_update(self.value_net.parameters(), self.target_net.parameters(), 0.005)

                success_rate = collected_step_summary['done']
                reward = collected_step_summary['reward']

                self.train_metrics.update('success_rate', success_rate)
                self.train_metrics.update('reward', reward)
                self.train_metrics.update('goal_distance', collected_step_summary['goal_distance'])
                self.train_metrics.update('jaw_state', collected_step_summary['jaw_state'])
                self.log_models_parameters()

                print("Epoch: {:4d}, Step: {:4d}, reward: {:3.2f}, success_rate: {:3.2f}".format(epoch, cycle,
                                                                                                 reward,
                                                                                                 success_rate))

            success_rate, reward, images = self._eval_agent(epoch)

            self.val_metrics.update('val/success_rate', success_rate)
            self.val_metrics.update('val/reward', reward)

            if self.should_save_episode_video and epoch % self.episode_video_timer == 0:
                imageio.mimsave(self.video_log_path.format(epoch), images)

            if self.should_save_model and epoch > 0 and epoch % self.model_save_timer == 0:
                epoch_path = self.model_path + "/epoch_" + str(epoch)
                create_dirs([epoch_path])
                self.transition_net.save(os.path.join(epoch_path, 'transition_net.pth'))
                self.actor.save(os.path.join(epoch_path, 'actor.pth'))
                self.value_net.save(os.path.join(epoch_path, 'value_net.pth'))

                if self.log_alpha is not None:
                    torch.save(self.log_alpha, os.path.join(epoch_path, 'log_alpha.pt'))

                if self.alpha_optimizer is not None:
                    torch.save({"state_dict": self.alpha_optimizer.state_dict()},
                               os.path.join(epoch_path, 'alpha_optimizer.pt'))

                if self.alpha_tensor is not None:
                    torch.save(self.alpha_tensor, os.path.join(epoch_path, 'alpha_tensor.pt'))

        self.env.close()
        print("Training finished at {}".format(datetime.now()))
        return success_rate, reward

    def log_models_parameters(self):
        # add histogram of model parameters to the tensorboard
        for name, p in self.transition_net.named_parameters():
            self.writer.add_histogram('transition_net_net_' + name, p, bins='auto')
        for name, p in self.actor.named_parameters():
            self.writer.add_histogram('actor_' + name, p, bins='auto')
        for name, p in self.value_net.named_parameters():
            self.writer.add_histogram('value_net_' + name, p, bins='auto')

    def _reset(self):
        self.train_metrics.reset()
        native_observation = self.env.reset()

        observation = native_observation['observation']
        achieved_goal = native_observation['achieved_goal']
        desired_goal = native_observation['desired_goal']

        if self.actor_action_distribution == 'StateDependentNoiseDistribution':
            self.actor.reset_noise()

        return observation, achieved_goal, desired_goal, False, 0

    def _select_action(self, input_tensor):
        with torch.no_grad():
            action = self.actor.predict(input_tensor)
            return action.cpu().numpy().flatten()

    def _preprocess_inputs(self, observation, goal):
        observation = self.o_norm.normalize(observation)
        goal = self.g_norm.normalize(goal)
        # concatenate the stuffs
        inputs = np.concatenate([observation, goal])
        return torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _preprocess_batch_inputs(self, observation_batch, goal_batch):
        observation_batch = self.o_norm.normalize(observation_batch)
        goal_batch = self.g_norm.normalize(goal_batch)
        # concatenate the stuffs
        inputs = np.concatenate([observation_batch, goal_batch], axis=1)
        return torch.tensor(inputs, dtype=torch.float32, device=self.device)

    def _update_normalizer(self, observation, achieved_goal, desired_goal, action, info):
        # get the number of normalization transitions
        num_transitions = action.shape[0]
        # create the new buffer to store them
        sequential_batches, reward_batch, done_batch = self.her_module.sample_her_transitions(observation,
                                                                                              achieved_goal,
                                                                                              desired_goal,
                                                                                              action,
                                                                                              np.expand_dims(info, -1),
                                                                                              num_transitions)

        (observation_batch, _, desired_goal_batch, _, _) = sequential_batches[0]

        # update
        self.o_norm.update(observation_batch)
        self.g_norm.update(desired_goal_batch)
        # self.a_norm.update(actions)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        # self.a_norm.recompute_stats()

    def warmup(self):
        collected_step_episodes = []
        for _ in range(self.n_warmap_episodes):

            episode_data = EpisodeData()
            observation, achieved_goal, desired_goal, done, reward = self._reset()

            for episode_step in range(self._max_episode_steps):
                input_tensor = self._preprocess_inputs(observation, desired_goal)
                action = self._select_action(input_tensor)

                # feed the actions into the environment
                new_observation, reward, _, info = self.env.step(action)

                episode_data.add(observation.copy(), achieved_goal.copy(), desired_goal.copy(), action.copy(),
                                 info)

                observation = new_observation['observation']
                achieved_goal = new_observation['achieved_goal']

            collected_step_episodes.append(episode_data)

        collected_step_episodes = EpisodeData.as_dict_of_numpy_arrays(collected_step_episodes)

        # store the episodes
        self.buffer.store_episode(**collected_step_episodes, n_episodes_to_store=self.n_warmap_episodes)
        self._update_normalizer(**collected_step_episodes)


def make_env(config):
    if config.render_mode == 'none':
        env = gym.make(config.env_id)
    else:
        env = gym.make(config.env_id, render_mode=config.render_mode)
    # env = Monitor(env, prepare_path(config.monitor_file, experiment_name=config.experiment_name))
    env.seed(config.seed)
    return env


def as_tensor(numpy_array, device):
    return torch.tensor(numpy_array, dtype=torch.float32, device=device)


def interpret_boolean(param):
    if type(param) == bool:
        return param
    elif param in ['True', 'true', '1']:
        return True
    elif param in ['False', 'false', '0']:
        return False
    else:
        sys.exit("param '{}' cannot be interpreted as boolean".format(param))


def get_env_parameters(env):
    # Get spaces parameters
    observation_dim = env.observation_space.spaces['observation'].shape[0]
    goal_dim = env.observation_space.spaces['desired_goal'].shape[0]

    action_dim = env.action_space.shape[0]
    max_action_value = float(env.action_space.high[0])

    return observation_dim, goal_dim, action_dim, max_action_value


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


def set_random_seed(seed: int, device: str = 'cpu') -> None:
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if device == 'cuda':
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def train_agent_according_config(config):
    env = make_env(config)
    set_random_seed(config.seed, config.device_id)
    print(f'Actions count: {env.action_space.shape}')
    print(f'Action UB:   {float(env.action_space.high[0])}')
    print(f'Action LB: {float(env.action_space.low[0])}')

    agent = Agent(env, config)
    return agent.train()


if __name__ == '__main__':
    train_agent_according_config(get_config(env_id='NeedleReach-v0', device='cpu'))