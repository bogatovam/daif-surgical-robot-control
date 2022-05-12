import copy
import math
import os.path
import random
import sys
from abc import abstractmethod, ABC
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import surrol.gym as surrol_gym
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from stable_baselines3.common.distributions import StateDependentNoiseDistribution, \
    TanhBijector, DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import obs_as_tensor
from torch.distributions import Normal

from base import BaseModel
from config.configs_reader import get_config
import gym

from config.const import ROOT_DIR_PATH
from logger import TensorboardWriter
from utils import MetricTracker

import threading
import numpy as np
from datetime import datetime
import os.path
import random
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import gym

import threading
import numpy as np

import os.path
import random
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import gym

import threading
import numpy as np

torch.autograd.set_detect_anomaly(True)


class ReplayBuffer:
    def __init__(self, env, max_episode_steps, buffer_size, sample_func, device):
        self.observation_dim, self.goal_dim, self.action_dim, self.action_max = get_env_parameters(env)
        self.device = device

        self.max_episode_steps = max_episode_steps
        self.size = buffer_size // self.max_episode_steps
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info

        self.observation_memory = np.empty([self.size, self.max_episode_steps, self.observation_dim], dtype=np.float32)
        self.achieved_goal_memory = np.empty([self.size, self.max_episode_steps, self.goal_dim], dtype=np.float32)
        self.desired_goal_memory = np.empty([self.size, self.max_episode_steps, self.goal_dim], dtype=np.float32)
        self.actions_memory = np.empty([self.size, self.max_episode_steps, self.action_dim], dtype=np.float32)
        self.log_prob_memory = np.empty([self.size, self.max_episode_steps, 1], dtype=np.float32)

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, observation, achieved_goal, desired_goal, action, log_prob, n_episodes_to_store):
        with self.lock:
            ids = self._get_storage_idx(inc=n_episodes_to_store)
            # store the information
            self.observation_memory[ids] = observation
            self.achieved_goal_memory[ids] = achieved_goal
            self.desired_goal_memory[ids] = desired_goal
            self.actions_memory[ids] = action
            self.log_prob_memory[ids] = log_prob

            self.n_transitions_stored += self.max_episode_steps * n_episodes_to_store

    # sample the data from the replay buffer
    def sample(self, batch_size):
        observation_buffer = self.observation_memory[:self.current_size]
        achieved_goal_buffer = self.achieved_goal_memory[:self.current_size]
        desired_goal_buffer = self.desired_goal_memory[:self.current_size]
        actions_buffer = self.actions_memory[:self.current_size]
        log_prob_buffer = self.log_prob_memory[:self.current_size]

        return self.sample_func(observation_buffer,
                                achieved_goal_buffer, desired_goal_buffer,
                                actions_buffer, log_prob_buffer,
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


class HERSampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None, done_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.done_func = done_func

        # When sampling from memory at index i, obs_indices indicates that we want observations
        # with indices i-obs_indices, works the same for the others
        self.obs_indices = [2, 1, 0]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1

    def sample_her_transitions(self, observation_buffer,
                               achieved_goal_buffer, desired_goal_buffer,
                               actions_buffer, log_prob_buffer, batch_size):
        # Trajectory length
        trajectory_length = actions_buffer.shape[1]

        # Buffer size
        buffer_length = actions_buffer.shape[0]

        # generate ids which trajectories to use
        episode_ids = np.random.randint(low=0, high=buffer_length - 2, size=batch_size)

        # generate ids which timestamps to use
        # - 2 because we sample for 3 sequential timestamps
        t_samples = np.random.randint(low=0, high=trajectory_length - 2, size=batch_size)

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # previous
        t0 = self._sample_for_time(observation_buffer, achieved_goal_buffer, desired_goal_buffer,
                                   actions_buffer, log_prob_buffer,
                                   episode_ids, t_samples, her_indexes,
                                   batch_size=batch_size, time=0)
        # current
        t1 = self._sample_for_time(observation_buffer, achieved_goal_buffer, desired_goal_buffer,
                                   actions_buffer, log_prob_buffer,
                                   episode_ids, t_samples, her_indexes,
                                   batch_size=batch_size, time=1)
        # next
        t2 = self._sample_for_time(observation_buffer, achieved_goal_buffer, desired_goal_buffer,
                                   actions_buffer, log_prob_buffer,
                                   episode_ids, t_samples, her_indexes,
                                   batch_size=batch_size, time=2)

        (_, achieved_goal_batch_t1, desired_goal_batch_t1, _, _) = t1
        # Recompute the reward for the augmented 'desired_goal'
        # todo use achieved_goal_batch_t2 and desired_goal_batch_t1
        reward_batch = self.reward_func(achieved_goal_batch_t1, desired_goal_batch_t1, info=None)
        # Recompute the termination state for the augmented 'desired_goal'
        done_batch = self.done_func(achieved_goal_batch_t1, desired_goal_batch_t1, info=None)

        # Reshape the batch
        reward_batch = reward_batch.reshape(batch_size, *reward_batch.shape[1:])
        done_batch = done_batch.reshape(batch_size, *done_batch.shape[1:])

        return t0, t1, t2, reward_batch, done_batch

    def _sample_for_time(self, observation_buffer, achieved_goal_buffer, desired_goal_buffer,
                         actions_buffer, log_prob_buffer,
                         episode_idxs, t_samples, her_indexes, batch_size, time):
        # Trajectory length
        trajectory_length = actions_buffer.shape[1] - 3

        observation_batch = observation_buffer[:, time:, :][episode_idxs, t_samples].copy()
        achieved_goal_batch = achieved_goal_buffer[:, time:, :][episode_idxs, t_samples].copy()
        desired_goal_batch = desired_goal_buffer[:, time:, :][episode_idxs, t_samples].copy()
        actions_batch = actions_buffer[:, time:, :][episode_idxs, t_samples].copy()
        log_prob_batch = log_prob_buffer[:, time:, :][episode_idxs, t_samples].copy()

        # Reshape the batch
        observation_batch = observation_batch.reshape(batch_size, *observation_batch.shape[1:])
        achieved_goal_batch = achieved_goal_batch.reshape(batch_size, *achieved_goal_batch.shape[1:])
        desired_goal_batch = desired_goal_batch.reshape(batch_size, *desired_goal_batch.shape[1:])
        actions_batch = actions_batch.reshape(batch_size, *actions_batch.shape[1:])
        log_prob_batch = log_prob_batch.reshape(batch_size, *log_prob_batch.shape[1:])

        # Sample 'future' timestamps for each 't_samples'
        future_offset = np.random.uniform(size=batch_size) * (trajectory_length - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]

        # Get the achieved_goal at the 'future' timestamps
        next_achieved_goal = achieved_goal_buffer[:, time:, :][episode_idxs[her_indexes], future_t]
        # Replace the 'desired_goal' with the 'next_achieved_goal'
        desired_goal_batch[her_indexes] = next_achieved_goal

        return observation_batch, achieved_goal_batch, desired_goal_batch, actions_batch, log_prob_batch


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
        self.set_training_mode(False)

        observation = self.obs_to_tensor(observation)

        with torch.no_grad():
            actions = self._predict(observation)

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions

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
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_tensor(observation, self.device)
        return observation


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super(SquashedDiagGaussianDistribution, self).__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(self, mean_actions: torch.Tensor,
                           log_std: torch.Tensor) -> "SquashedDiagGaussianDistribution":
        super(SquashedDiagGaussianDistribution, self).proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super(SquashedDiagGaussianDistribution, self).log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return torch.tanh(self.gaussian_actions)

    def mode(self) -> torch.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return torch.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class Actor(BasePolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            input_size,
            net_arch,
            weight_decay=0.00001,
            lr=0.001,
            activation_fn=nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            full_std: bool = True,
            sde_net_arch=None,
            use_expln: bool = True,
            clip_mean: float = 1.0,
            normalize_images: bool = True,
    ):
        super().__init__(observation_space, action_space, normalize_images=normalize_images, squash_output=True)

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        action_dim = get_action_dim(self.action_space)
        # here will be vae
        self.latent_pi = nn.Sequential(*create_mlp(input_size, -1, net_arch, activation_fn))

        last_layer_dim = net_arch[-1]

        self.action_dist = DiagGaussianDistribution(action_dim)
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))

        self.log_std = nn.Linear(last_layer_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                clip_mean=self.clip_mean,
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

        if self.use_sde:
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
    def __init__(self, input_size, layer_sizes, output_size, lr=0.001, output_activation=torch.nn.Identity,
                 activation=torch.nn.SiLU, weight_decay=1e-4, device='cpu'):
        super(MLP, self).__init__()
        sizes = [input_size] + layer_sizes + [output_size]

        self.layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            self.layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]

        self.layers = nn.ModuleList(self.layers)
        self.optimizer = optim.Adam(self.parameters(), lr, weight_decay=weight_decay)  # Adam optimizer

        self.device = device
        self.to(self.device)

    def forward(self, inp):
        x = inp
        for layer in self.layers:
            x = layer(x)
        return x


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


class Agent:
    def __init__(self, env, config):
        self.env = env
        self.experiment_name = config.experiment_name
        self.experiment_description = config.experiment_description
        self.observation_dim, self.goal_dim, self.action_dim, self.action_max = get_env_parameters(env)

        self.device = config.device_id

        self.polyak = int(config.polyak)

        self.n_epochs = int(config.n_epochs)
        self.n_cycles = int(config.n_cycles)
        self.n_episodes = int(config.n_episodes)
        self.n_batches = int(config.n_batches)
        self._max_episode_steps = int(config.max_episode_steps)

        self.batch_size = int(config.batch_size)
        self.memory_capacity = int(config.memory_capacity)

        self.freeze_period = int(config.freeze_period)

        self.gamma = float(config.gamma)  # A precision parameter
        self.beta = float(config.beta)  # The discount rate
        self.noise_eps = float(config.noise_eps)  # The discount rate
        self.random_eps = float(config.random_eps)  # The discount rate
        self.random_eps = float(config.random_eps)  # The discount rate

        self.print_timer = int(config.print_timer)

        self.should_save_model = interpret_boolean(config.should_save_model)
        self.model_path = prepare_path(config.model_path, experiment_name=config.experiment_name)
        self.final_model_path = os.path.join(self.model_path, "final")
        self.model_save_timer = int(config.model_save_timer)

        self.should_save_episode_video = interpret_boolean(config.should_save_episode_video)
        self.episode_video_timer = int(config.episode_video_timer)

        self.state_shape = np.add(self.env.observation_space['observation'].shape,
                                  self.env.observation_space['desired_goal'].shape)
        self.state_size = np.prod(self.state_shape)

        self.actions_shape = self.env.action_space.shape
        self.action_dim = self.env.action_space.shape[-1]

        self.n_sampled_actions = config.n_sampled_actions
        self.freeze_cntr = 0

        self.n_screens = 4
        self.n_latent_states = 32
        self.lr_vae = 1e-5

        self.vae = VAE(self.n_screens, self.n_latent_states, lr=self.lr_vae, device=self.device)

        self.actor = Actor(env.observation_space, env.action_space,
                           self.state_size,
                           [128, 256, 128])

        self.transition_net = MLP(self.state_size + self.action_dim,
                                  [128, 256, 128], self.state_size, lr=0.001,
                                  device=self.device)

        self.value_net = MLP(self.state_size + self.action_dim,
                             [128, 256, 128],
                             1, device=self.device)
        self.target_net = MLP(self.state_size + self.action_dim,
                              [128, 256, 128],
                              1, device=self.device)

        self.her_module = HERSampler(config.replay_strategy, config.replay_k, self.env.compute_reward,
                                     self.env.compute_reward)
        # create the replay buffer
        self.buffer = ReplayBuffer(self.env, config.max_episode_steps, self.memory_capacity,
                                   self.her_module.sample_her_transitions, config.device_id)

        self.o_norm = Normalizer(size=env.observation_space.spaces['observation'].shape[0])
        self.g_norm = Normalizer(size=env.observation_space.spaces['desired_goal'].shape[0])
        self.a_norm = Normalizer(size=self.action_dim)
        self.target_update_interval = 2

        self.writer = TensorboardWriter(prepare_path(config.tb_log_folder, experiment_name=config.experiment_name),
                                        True)

        self.metrics = MetricTracker('vfe', 'efe_mse_loss', 'success_rate', 'reward', 'elapsed_steps_count',
                                     'transition_net_grad', 'actor_grad_acc', 'value_net_grad', writer=self.writer)

    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        t0, t1, t2, reward_batch, done_batch = self.buffer.sample(self.batch_size)

        # Retrieve a batch for 3 consecutive points in time
        (observation_batch_t0, achieved_goal_batch_t0, desired_goal_batch_t0, actions_batch_t0, log_prob_batch_t0) = t0
        (observation_batch_t1, achieved_goal_batch_t1, desired_goal_batch_t1, actions_batch_t1, log_prob_batch_t1) = t1
        (observation_batch_t2, achieved_goal_batch_t2, desired_goal_batch_t2, actions_batch_t2, log_prob_batch_t2) = t2

        state_batch_t0 = self._preprocess_batch_inputs(observation_batch_t0, desired_goal_batch_t0)
        state_batch_t1 = self._preprocess_batch_inputs(observation_batch_t1, desired_goal_batch_t1)
        state_batch_t2 = self._preprocess_batch_inputs(observation_batch_t2, desired_goal_batch_t2)

        actions_batch_t0 = self.as_tensor(actions_batch_t0)
        actions_batch_t1 = self.as_tensor(actions_batch_t1)
        actions_batch_t2 = self.as_tensor(actions_batch_t2)

        log_prob_batch_t0 = self.as_tensor(log_prob_batch_t0)
        log_prob_batch_t1 = self.as_tensor(log_prob_batch_t1)
        log_prob_batch_t2 = self.as_tensor(log_prob_batch_t2)

        reward_batch, done_batch = self.as_tensor(reward_batch), self.as_tensor(done_batch)

        # At time t0 predict the state at time t1:
        # append actions vector nearby state
        X = torch.cat((state_batch_t0, actions_batch_t0), dim=1)
        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
            pred_batch_t0t1, state_batch_t1, reduction='none'), dim=1).unsqueeze(1)

        return (state_batch_t0, state_batch_t1, state_batch_t2,
                actions_batch_t0, actions_batch_t1, actions_batch_t2,
                log_prob_batch_t0, log_prob_batch_t1, log_prob_batch_t2,
                reward_batch, done_batch, pred_error_batch_t0t1)

    def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
                               actions_batch_t1, actions_batch_t2,
                               log_prob_batch_t1, log_prob_batch_t2,
                               reward_batch, done_batch, pred_error_batch_t0t1):

        with torch.no_grad():
            targe_net_input = torch.cat([state_batch_t2, actions_batch_t2], dim=1)
            target_expected_free_energies_batch_t2 = self.target_net(targe_net_input)
            action_probs_t2 = torch.exp(log_prob_batch_t2).clamp(0, 1)

            # Weigh the target EFEs according to the action distribution:
            weighted_targets = (action_probs_t2 * target_expected_free_energies_batch_t2).sum(-1).unsqueeze(1)

            # Determine the batch of bootstrapped estimates of the EFEs:
            expected_free_energy_estimate_batch = (
                    -reward_batch + pred_error_batch_t0t1 + self.beta * weighted_targets)

        # Determine the Expected free energy at time t1 according to the value network:
        value_net_input = torch.cat([state_batch_t1, actions_batch_t1], dim=1)
        value_net_output = self.value_net(value_net_input)
        action_probs_t1 = torch.exp(log_prob_batch_t1).clamp(0, 1)

        efe_batch_t1 = (value_net_output)

        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(expected_free_energy_estimate_batch, efe_batch_t1)
        return value_net_loss

    def compute_variational_free_energy(self, state_batch_t1, pred_error_batch_t0t1):
        predicted_actions_t1, pred_log_prob_t1 = self.actor.action_log_prob(state_batch_t1)
        action_probs_t1 = torch.exp(pred_log_prob_t1).clamp(0, 1).unsqueeze(1)

        value_net_input = torch.cat([state_batch_t1, predicted_actions_t1], dim=1)

        expected_free_energy_t1 = self.value_net(value_net_input)

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_expected_free_energy_batch_t1 = torch.sigmoid(-self.gamma * expected_free_energy_t1)
        boltzmann_expected_free_energy_batch_t1 = boltzmann_expected_free_energy_batch_t1.clamp(min=1e-9, max=1 - 1e-9)

        # Weigh them according to the action distribution:
        energy_batch = -(action_probs_t1 * torch.log(boltzmann_expected_free_energy_batch_t1)).view(self.batch_size, 1)

        # Determine the entropy of the action distribution
        entropy_batch = -(action_probs_t1 * torch.log(action_probs_t1)).view(self.batch_size, 1)

        # Determine the Variable Free Energy, then take the mean over all batch samples:
        vfe_batch = pred_error_batch_t0t1 + energy_batch - entropy_batch
        vfe = torch.mean(vfe_batch)
        return vfe

    def _update_network(self):
        # Retrieve transition data in mini batches:
        (state_batch_t0, state_batch_t1, state_batch_t2,
         actions_batch_t0, actions_batch_t1, actions_batch_t2,
         log_prob_batch_t0, log_prob_batch_t1, log_prob_batch_t2,
         reward_batch, done_batch, pred_error_batch_t0t1) = self.get_mini_batches()
        # Compute the value network loss:

        value_net_loss = self.compute_value_net_loss(state_batch_t1, state_batch_t2,
                                                     actions_batch_t1, actions_batch_t2,
                                                     log_prob_batch_t1, log_prob_batch_t2,
                                                     reward_batch, done_batch, pred_error_batch_t0t1)

        # Determine the reconstruction loss for time t1
        recon_batch = self.vae.decode(z_batch_t1, self.batch_size)
        vae_loss = self.vae.loss_function(recon_batch, obs_batch_t1, state_mu_batch_t1, state_logvar_batch_t1,
                                          batch=True) / self.alpha

        self.transition_net.optimizer.zero_grad()

        self.value_net.optimizer.zero_grad()
        value_net_loss.backward()

        value_net_grad = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 100.)

        self.value_net.optimizer.step()

        # Compute the variational free energy:
        vfe = self.compute_variational_free_energy(state_batch_t1, pred_error_batch_t0t1)

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        vfe.backward()

        actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.)
        transition_net_grad = torch.nn.utils.clip_grad_norm_(self.transition_net.parameters(), 100.)

        self.actor.optimizer.step()
        self.transition_net.optimizer.step()

        return vfe.item(), value_net_loss.item(), transition_net_grad.item(), actor_grad.item(), value_net_grad.item()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

    # do the evaluation
    def _eval_agent(self, epoch):
        images = []
        total_reward = []
        total_success_rate = []
        elapsed_steps_count = []

        for cycle in range(epoch):
            per_reward = []
            observation, _, desired_goal, _, reward = self._reset()

            done = False
            episode_step = 0
            while not done and episode_step < self._max_episode_steps:
                input_tensor = self._preprocess_inputs(observation, desired_goal)
                action = self._select_action(input_tensor)

                # feed the actions into the environment
                new_observation, reward, done, info = self.env.step(action)

                observation = new_observation['observation']
                per_reward.append(reward)
                episode_step += 1

                # if self.should_save_episode_video and epoch % self.episode_video_timer == 0:
                #     images += [self.env.render()]

            total_reward.append(np.mean(per_reward))
            total_success_rate.append(1 if done and episode_step < self._max_episode_steps else 0)
            elapsed_steps_count.append(episode_step)

        total_reward = np.asarray(total_reward)
        total_success_rate = np.asarray(total_success_rate)
        elapsed_steps_count = np.asarray(elapsed_steps_count)
        return np.mean(total_success_rate), np.mean(total_reward), np.asarray(images), np.mean(elapsed_steps_count)

    def train(self):
        self.writer.add_text(self.experiment_name, self.experiment_description)
        print("Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.now()))

        for epoch in range(self.n_epochs):
            for step in range(self.n_cycles):
                self.writer.set_step(self.n_cycles * epoch + step)

                cycle_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [],
                              'action': [], 'log_prob': []}

                cycle_summary_data = {'done': [], 'reward': []}

                for _ in range(self.n_episodes):
                    observation, achieved_goal, desired_goal, done, reward = self._reset()

                    episode_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [],
                                    'action': [], 'log_prob': []}

                    episode_summary_data = {'done_step': self._max_episode_steps, 'reward': []}
                    for episode_step in range(self._max_episode_steps):
                        input_tensor = self._preprocess_inputs(observation, desired_goal)
                        action, log_prob = self._select_action(input_tensor)

                        episode_data['observation'].append(observation.copy())
                        episode_data['achieved_goal'].append(achieved_goal.copy())
                        episode_data['desired_goal'].append(desired_goal.copy())
                        episode_data['action'].append(action.copy())
                        episode_data['log_prob'].append(log_prob.copy())

                        # feed the actions into the environment
                        new_observation, reward, done, _ = self.env.step(action)
                        obs = self.get_screen(self.env, self.device)

                        if done:
                            episode_summary_data['done_step'] = episode_step
                        episode_summary_data['reward'].append(np.mean(reward))

                        observation = new_observation['observation']
                        achieved_goal = new_observation['achieved_goal']

                    cycle_data['observation'].append(np.asarray(episode_data['observation'], dtype=np.float32))
                    cycle_data['achieved_goal'].append(np.asarray(episode_data['achieved_goal'], dtype=np.float32))
                    cycle_data['desired_goal'].append(np.asarray(episode_data['desired_goal'], dtype=np.float32))
                    cycle_data['action'].append(np.asarray(episode_data['action'], dtype=np.float32))
                    cycle_data['log_prob'].append(np.asarray(episode_data['log_prob'], dtype=np.float32))

                    cycle_summary_data['done'].append(episode_summary_data['done_step'] < self._max_episode_steps)
                    cycle_summary_data['reward'].append(np.mean(episode_summary_data['reward']))

                cycle_data['observation'] = np.asarray(cycle_data['observation'], dtype=np.float32)
                cycle_data['achieved_goal'] = np.asarray(cycle_data['achieved_goal'], dtype=np.float32)
                cycle_data['desired_goal'] = np.asarray(cycle_data['desired_goal'], dtype=np.float32)
                cycle_data['action'] = np.asarray(cycle_data['action'], dtype=np.float32)
                cycle_data['log_prob'] = np.asarray(cycle_data['log_prob'], dtype=np.float32)

                cycle_summary_data['done'] = np.asarray(cycle_summary_data['done'], dtype=np.float32)
                cycle_summary_data['reward'] = np.asarray(cycle_summary_data['reward'], dtype=np.float32)

                # store the episodes
                self.buffer.store_episode(**cycle_data, n_episodes_to_store=self.n_episodes)
                self._update_normalizer(**cycle_data)

                vfe = []
                value_net_loss = []
                (transition_net_grad_acc, actor_grad_acc, value_net_grad_acc) = 0, 0, 0
                for _ in range(self.n_batches):
                    # train the network
                    (vfe_item, value_net_loss_item, transition_net_grad, actor_grad,
                     value_net_grad) = self._update_network()
                    vfe.append(vfe_item), value_net_loss.append(value_net_loss_item)

                    transition_net_grad_acc += transition_net_grad
                    actor_grad_acc += actor_grad
                    value_net_grad_acc += value_net_grad

                self.metrics.update('vfe', np.mean(vfe))
                self.metrics.update('efe_mse_loss', np.mean(value_net_loss))
                self.metrics.update('transition_net_grad', transition_net_grad_acc / self.n_batches)
                self.metrics.update('actor_grad_acc', actor_grad_acc / self.n_batches)
                self.metrics.update('value_net_grad', value_net_grad_acc / self.n_batches)

                # soft update
                if step % self.target_update_interval == 0:
                    self._soft_update_target_network(self.target_net, self.value_net)
                # val_success_rate, val_reward, images, elapsed_steps_count = self._eval_agent(10)
                val_success_rate = np.mean(cycle_summary_data['done'])
                val_reward = np.mean(cycle_summary_data['reward'])

                self.metrics.update('success_rate', val_success_rate)
                self.metrics.update('reward', val_reward)
                self.log_models_parameters()

                print(
                    "Epoch: {:4d}, Step: {:4d}, reward: {:3.2f}, success_rate: {:3.2f}".format(epoch, step, val_reward,
                                                                                               val_success_rate))

                # plot_grad_flow(self.prediction_policy_mu_network.named_parameters())

            # if self.should_save_episode_video and epoch % self.episode_video_timer == 0:
            #     self.writer.add_video(self.experiment_name + f'_{epoch}', images)

            if self.should_save_model and epoch > 0 and epoch % self.model_save_timer == 0:
                self.transition_net.save(os.path.join(self.model_path, 'transition_net.pth'))
                self.actor.save(os.path.join(self.model_path, 'actor.pth'))
                self.value_net.save(os.path.join(self.model_path, 'value_net.pth'))

        self.env.close()

        if self.should_save_model:
            self.transition_net.save(os.path.join(self.final_model_path, 'transition_net.pth'))
            self.actor.save(os.path.join(self.final_model_path, 'actor.pth'))
            self.value_net.save(os.path.join(self.final_model_path, 'value_net.pth'))

        # Print and keep a (.txt) record of stuff
        print("Training finished at {}".format(datetime.now()))

    def log_models_parameters(self):
        # add histogram of model parameters to the tensorboard
        for name, p in self.transition_net.named_parameters():
            self.writer.add_histogram('transition_net_net_' + name, p, bins='auto')
        for name, p in self.actor.named_parameters():
            self.writer.add_histogram('actor_' + name, p, bins='auto')
        for name, p in self.value_net.named_parameters():
            self.writer.add_histogram('value_net_' + name, p, bins='auto')

    def _reset(self):
        self.metrics.reset()
        native_observation = self.env.reset()

        observation = native_observation['observation']
        achieved_goal = native_observation['achieved_goal']
        desired_goal = native_observation['desired_goal']

        # observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        # achieved_goal = torch.tensor(achieved_goal, dtype=torch.float32, device=self.device)
        # desired_goal = torch.tensor(desired_goal, dtype=torch.float32, device=self.device)

        return observation, achieved_goal, desired_goal, False, 0

    def _select_action(self, input_tensor):
        with torch.no_grad():
            action, log_prob = self.actor.action_log_prob(input_tensor)
            # action = torch.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
            return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def _sample_actions_with_probs(self, input_tensor, n_samples):
        # input_tensor.shape = (batch_size, obs_dim + desired_goal)
        # Determine the action distribution given the current observation:
        mu, log_variance = self.prediction(input_tensor)
        # mu.shape = (batch_size, action_dim)

        r_mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        r_log_variance = log_variance.unsqueeze(1).repeat(1, n_samples, 1)
        # r_mu.shape = (batch_size, n_samples, action_dim)

        actions = self._infer_action_using_reparameterization(r_mu, r_log_variance).detach()
        # actions.shape = (batch_size, n_samples, action_dim)
        probs = torch.exp((self._log_prob(actions, r_mu, self._variance2std(r_log_variance))))
        # actions = self.a_norm.normalize(actions)
        return mu, log_variance, actions, probs

    def _infer_action_using_reparameterization(self, mu, log_variance):
        # Apply reparameterization trick
        std = self._variance2std(log_variance)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _variance2std(log_variance):
        return torch.exp(0.5 * log_variance)

    @staticmethod
    def _log_prob(value, loc, scale):
        # compute the variance
        var = torch.pow(scale, 2)
        log_scale = torch.log(scale)
        return -((value.detach() - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def _preprocess_inputs(self, observation, goal):
        # observation = self.o_norm.normalize(observation)
        # goal = self.g_norm.normalize(goal)
        # concatenate the stuffs
        inputs = np.concatenate([observation, goal])
        return torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _preprocess_batch_inputs(self, observation_batch, goal_batch):
        # observation_batch = self.o_norm.normalize(observation_batch)
        # goal_batch = self.g_norm.normalize(goal_batch)
        # concatenate the stuffs
        inputs = np.concatenate([observation_batch, goal_batch], axis=1)
        return torch.tensor(inputs, dtype=torch.float32, device=self.device)

    def _update_normalizer(self, observation, achieved_goal, desired_goal, action, log_prob):
        # get the number of normalization transitions
        num_transitions = action.shape[0]
        # create the new buffer to store them
        t0, t1, t2, reward_batch, done_batch = self.her_module.sample_her_transitions(observation, achieved_goal,
                                                                                      desired_goal,
                                                                                      action, log_prob,
                                                                                      num_transitions)

        (observation_batch, _, desired_goal_batch, actions, _) = t0

        # update
        self.o_norm.update(observation_batch)
        self.g_norm.update(desired_goal_batch)
        self.a_norm.update(actions)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.a_norm.recompute_stats()

    def as_tensor(self, numpy_array):
        return torch.tensor(numpy_array, dtype=torch.float32, device=self.device)


def make_env(config):
    env = gym.make(config.env_id, render_mode=config.render_mode)
    # env = Monitor(env, prepare_path(config.monitor_file, experiment_name=config.experiment_name))
    env.seed(config.seed)
    return env


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


def prepare_path(path, **args):
    return os.path.join(ROOT_DIR_PATH, path.format(**args))


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
    agent.train()


if __name__ == '__main__':
    train_agent_according_config(get_config(env_id='NeedleReach-v0', device='cpu'))
