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
from stable_baselines3.common.utils import polyak_update

import surrol.gym as surrol_gym
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from stable_baselines3.common.distributions import StateDependentNoiseDistribution, \
    TanhBijector, DiagGaussianDistribution, SquashedDiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose
from stable_baselines3.common.torch_layers import create_mlp

import utils
from base import BaseModel
from config.configs_reader import get_config
import gym

from config.const import ROOT_DIR_PATH
from logger import TensorboardWriter
from utils import MetricTracker, create_dirs

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
import imageio


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

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, observation, achieved_goal, desired_goal, action, n_episodes_to_store):
        with self.lock:
            ids = self._get_storage_idx(inc=n_episodes_to_store)
            # store the information
            self.observation_memory[ids] = observation
            self.achieved_goal_memory[ids] = achieved_goal
            self.desired_goal_memory[ids] = desired_goal
            self.actions_memory[ids] = action

            self.n_transitions_stored += self.max_episode_steps * n_episodes_to_store

    # sample the data from the replay buffer
    def sample(self, batch_size):
        observation_buffer = self.observation_memory[:self.current_size]
        achieved_goal_buffer = self.achieved_goal_memory[:self.current_size]
        desired_goal_buffer = self.desired_goal_memory[:self.current_size]
        actions_buffer = self.actions_memory[:self.current_size]

        return self.sample_func(observation_buffer,
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


class HERSampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, observation_buffer,
                               achieved_goal_buffer, desired_goal_buffer,
                               actions_buffer, batch_size):
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
        t0 = self._sample_for_time(observation_buffer, achieved_goal_buffer, desired_goal_buffer, actions_buffer,
                                   episode_ids, t_samples, her_indexes,
                                   batch_size=batch_size, time=0)
        # current
        t1 = self._sample_for_time(observation_buffer, achieved_goal_buffer, desired_goal_buffer, actions_buffer,
                                   episode_ids, t_samples, her_indexes,
                                   batch_size=batch_size, time=1)
        # next
        t2 = self._sample_for_time(observation_buffer, achieved_goal_buffer, desired_goal_buffer, actions_buffer,
                                   episode_ids, t_samples, her_indexes,
                                   batch_size=batch_size, time=2)

        (_, achieved_goal_batch_t1, desired_goal_batch_t1, _) = t1

        # Recompute the reward for the augmented 'desired_goal'
        # todo use achieved_goal_batch_t2 and desired_goal_batch_t1?
        reward_batch = self.reward_func(achieved_goal_batch_t1, desired_goal_batch_t1, info=None)
        # Recompute the termination state for the augmented 'desired_goal'
        done_batch = reward_batch == 0

        # Reshape the batch
        reward_batch = reward_batch.reshape(batch_size, *reward_batch.shape[1:])
        done_batch = done_batch.reshape(batch_size, *done_batch.shape[1:])

        return t0, t1, t2, reward_batch, done_batch

    def _sample_for_time(self, observation_buffer, achieved_goal_buffer, desired_goal_buffer, actions_buffer,
                         episode_idxs, t_samples, her_indexes, batch_size, time):
        # Trajectory length
        trajectory_length = actions_buffer.shape[1] - 3

        observation_batch = observation_buffer[:, time:, :][episode_idxs, t_samples].copy()
        achieved_goal_batch = achieved_goal_buffer[:, time:, :][episode_idxs, t_samples].copy()
        desired_goal_batch = desired_goal_buffer[:, time:, :][episode_idxs, t_samples].copy()
        actions_batch = actions_buffer[:, time:, :][episode_idxs, t_samples].copy()

        # Reshape the batch
        observation_batch = observation_batch.reshape(batch_size, *observation_batch.shape[1:])
        achieved_goal_batch = achieved_goal_batch.reshape(batch_size, *achieved_goal_batch.shape[1:])
        desired_goal_batch = desired_goal_batch.reshape(batch_size, *desired_goal_batch.shape[1:])
        actions_batch = actions_batch.reshape(batch_size, *actions_batch.shape[1:])

        # Sample 'future' timestamps for each 't_samples'
        future_offset = np.random.uniform(size=batch_size) * (trajectory_length - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]

        # Get the achieved_goal at the 'future' timestamps
        next_achieved_goal = achieved_goal_buffer[:, time:, :][episode_idxs[her_indexes], future_t]
        # Replace the 'desired_goal' with the 'next_achieved_goal'
        desired_goal_batch[her_indexes] = next_achieved_goal

        return observation_batch, achieved_goal_batch, desired_goal_batch, actions_batch


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

        # if not isinstance(observation, dict):
        #     # Add batch dimension if needed
        #     observation = observation.reshape((-1,) + self.observation_space.shape)

        return observation


class Actor(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            input_size,
            net_arch,
            weight_decay=0.00001,
            lr=0.0001,
            activation_fn=nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -1,
            full_std: bool = True,
            sde_net_arch=None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
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

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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
    def __init__(self, input_size, layer_sizes, output_size, lr=0.0001, output_activation=torch.nn.Identity,
                 activation=torch.nn.ReLU, weight_decay=1e-6, device='cpu'):
        super(MLP, self).__init__()

        self.layers = create_mlp(input_size, output_size, layer_sizes, activation)
        self.layers = nn.ModuleList(self.layers)
        self.optimizer = optim.Adam(self.parameters(), lr)  # Adam optimizer

        self.device = device
        self.to(self.device)

    def forward(self, inp):
        x = inp
        for layer in self.layers:
            x = layer(x)
        return x


class Agent:
    def __init__(self, env, config):
        self.env = env
        self.experiment_name = config.experiment_name
        self.experiment_description = config.experiment_description
        self.observation_dim, self.goal_dim, self.action_dim, self.action_max = get_env_parameters(env)

        self.device = config.device_id

        self.polyak = int(config.hparams.polyak)

        self.use_sde = int(config.hparams.use_sde)
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
        self.final_model_path = os.path.join(self.model_path, "final")
        self.model_save_timer = int(config.model_save_timer)

        self.should_save_episode_video = interpret_boolean(config.should_save_episode_video)
        self.episode_video_timer = int(config.episode_video_timer)

        self.state_shape = np.add(self.env.observation_space['observation'].shape,
                                  self.env.observation_space['desired_goal'].shape)
        self.state_size = np.prod(self.state_shape)

        self.actions_shape = self.env.action_space.shape
        self.action_dim = self.env.action_space.shape[-1]

        self.actor = Actor(env.observation_space, env.action_space,
                           self.state_size,
                           OmegaConf.to_object(config.hparams.actor_layers),
                           use_sde=config.hparams.use_sde,
                           lr=config.hparams.actor_lr)

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

        # Target entropy is used when learning the entropy coefficient
        if isinstance(self.alpha, str) and self.alpha.startswith("auto"):
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.alpha, str) and self.alpha.startswith("auto"):
            init_value = 1.0
            if "_" in self.alpha:
                init_value = float(self.alpha.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.hparams.alpha_lr)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.alpha_tensor = torch.tensor(float(self.alpha)).to(self.device)

        self.her_module = HERSampler(config.hparams.replay_strategy, config.hparams.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = ReplayBuffer(self.env, self._max_episode_steps, self.memory_capacity,
                                   self.her_module.sample_her_transitions, config.device_id)

        self.o_norm = Normalizer(size=env.observation_space.spaces['observation'].shape[0])
        self.g_norm = Normalizer(size=env.observation_space.spaces['desired_goal'].shape[0])
        self.a_norm = Normalizer(size=self.action_dim)
        self.target_update_interval = 1

        self.writer = TensorboardWriter(prepare_path(config.tb_log_folder, experiment_name=config.experiment_name),
                                        True)

        self.train_metrics = MetricTracker('vfe', 'efe_mse_loss', 'success_rate', 'reward',
                                           'transition_net_grad', 'actor_grad_acc', 'value_net_grad',
                                           'sde_std', 'alpha', 'predicted_log_prob_t1', 'min_qf_pi', 'next_q_values',
                                           'next_log_prob', writer=self.writer)

        self.val_metrics = MetricTracker('success_rate', 'reward', writer=self.writer)

        # just to save model configuration to logs
        config_as_dict = OmegaConf.to_object(config.hparams)
        config_as_dict['actor_layers'] = str(config_as_dict['actor_layers'])
        config_as_dict['transition_net_layers'] = str(config_as_dict['transition_net_layers'])
        config_as_dict['value_net_layers'] = str(config_as_dict['value_net_layers'])
        config_as_dict['max_episode_steps'] = str(env._max_episode_steps)

        with open(os.path.join(self.model_path, "config.yaml"), 'w+') as file:
            OmegaConf.save(config, file)

        # self.writer.add_hparams(config_as_dict, {}, run_name=config.experiment_name)

    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        t0, t1, t2, reward_batch, done_batch = self.buffer.sample(self.batch_size)

        # Retrieve a batch for 3 consecutive points in time
        (observation_batch_t0, achieved_goal_batch_t0, desired_goal_batch_t0, actions_batch_t0) = t0
        (observation_batch_t1, achieved_goal_batch_t1, desired_goal_batch_t1, actions_batch_t1) = t1
        (observation_batch_t2, achieved_goal_batch_t2, desired_goal_batch_t2, actions_batch_t2) = t2

        state_batch_t0 = self._preprocess_batch_inputs(observation_batch_t0, desired_goal_batch_t0)
        state_batch_t1 = self._preprocess_batch_inputs(observation_batch_t1, desired_goal_batch_t1)
        state_batch_t2 = self._preprocess_batch_inputs(observation_batch_t2, desired_goal_batch_t2)

        actions_batch_t0 = self.as_tensor(actions_batch_t0)
        actions_batch_t1 = self.as_tensor(actions_batch_t1)
        actions_batch_t2 = self.as_tensor(actions_batch_t2)

        reward_batch, done_batch = self.as_tensor(reward_batch), self.as_tensor(done_batch)

        return (state_batch_t0, state_batch_t1, state_batch_t2,
                actions_batch_t0, actions_batch_t1, actions_batch_t2,
                reward_batch, done_batch)

    def _update_network(self):
        # stable-baseline-3 implementation
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        # Retrieve transition data in mini batches:
        (state_batch_t0, state_batch_t1, state_batch_t2,
         actions_batch_t0, actions_batch_t1, actions_batch_t2,
         reward_batch, done_batch) = self.get_mini_batches()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(state_batch_t1)
        log_prob = log_prob.reshape(-1, 1)

        alpha_loss = None
        if self.alpha_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha = torch.exp(self.log_alpha.detach())
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        else:
            alpha = self.alpha_tensor

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(state_batch_t2)
            # Compute the next Q values: min over all critics targets
            next_q_values = self.target_net(torch.cat([state_batch_t2, next_actions], dim=1))
            # add entropy term
            next_q_values = next_q_values - alpha * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = reward_batch.reshape(-1, 1) + (
                    1.0 - done_batch.reshape(-1, 1)) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer

        current_q_values = self.value_net(torch.cat([state_batch_t1, actions_batch_t1], dim=1))

        # Compute critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the critic
        self.value_net.optimizer.zero_grad()
        critic_loss.backward()
        self.value_net.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        min_qf_pi = self.value_net(torch.cat([state_batch_t1, actions_pi], dim=1))
        actor_loss = (alpha * log_prob - min_qf_pi).mean()

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        polyak_update(self.value_net.parameters(), self.target_net.parameters(), 0.005)

        metrics = dict(
            vfe=actor_loss.item(),
            efe_mse_loss=critic_loss.item(),
            alpha=alpha.item().detach().item(),
            predicted_log_prob_t1=log_prob.mean().detach().item(),
            min_qf_pi=min_qf_pi.mean().detach().item(),
            next_q_values=next_q_values.mean().detach().item(),
            next_log_prob=next_log_prob.mean().detach().item(),
        )
        return metrics

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
        for epoch in range(self.n_epochs):
            for cycle in range(self.steps_per_epoch):
                step = self.steps_per_epoch * epoch + cycle
                self.writer.set_step(step)

                cycle_summary_data = {'done': [], 'reward': []}
                cycle_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [], 'action': []}

                for _ in range(self.n_rollout_episodes):
                    observation, achieved_goal, desired_goal, done, reward = self._reset()

                    episode_summary_data = {'done': [], 'reward': []}
                    episode_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [], 'action': []}

                    for episode_step in range(self._max_episode_steps):
                        input_tensor = self._preprocess_inputs(observation, desired_goal)
                        action = self._select_action(input_tensor)

                        episode_data['observation'].append(observation.copy())
                        episode_data['achieved_goal'].append(achieved_goal.copy())
                        episode_data['desired_goal'].append(desired_goal.copy())
                        episode_data['action'].append(action.copy())

                        # feed the actions into the environment
                        new_observation, reward, _, info = self.env.step(action)

                        episode_summary_data['reward'].append(np.mean(reward))
                        episode_summary_data['done'].append(info['is_success'])

                        observation = new_observation['observation']
                        achieved_goal = new_observation['achieved_goal']

                    cycle_data['observation'].append(np.asarray(episode_data['observation'], dtype=np.float32))
                    cycle_data['achieved_goal'].append(np.asarray(episode_data['achieved_goal'], dtype=np.float32))
                    cycle_data['desired_goal'].append(np.asarray(episode_data['desired_goal'], dtype=np.float32))
                    cycle_data['action'].append(np.asarray(episode_data['action'], dtype=np.float32))

                    cycle_summary_data['done'].append(np.mean(episode_summary_data['done']))
                    cycle_summary_data['reward'].append(np.mean(episode_summary_data['reward']))

                cycle_data['observation'] = np.asarray(cycle_data['observation'], dtype=np.float32)
                cycle_data['achieved_goal'] = np.asarray(cycle_data['achieved_goal'], dtype=np.float32)
                cycle_data['desired_goal'] = np.asarray(cycle_data['desired_goal'], dtype=np.float32)
                cycle_data['action'] = np.asarray(cycle_data['action'], dtype=np.float32)

                cycle_summary_data['done'] = np.asarray(cycle_summary_data['done'], dtype=np.float32)
                cycle_summary_data['reward'] = np.asarray(cycle_summary_data['reward'], dtype=np.float32)

                # store the episodes
                self.buffer.store_episode(**cycle_data, n_episodes_to_store=self.n_rollout_episodes)
                self._update_normalizer(**cycle_data)

                train_iteration_metrics = []
                for _ in range(self.n_training_iterations):
                    # train the network
                    metrics_dict = self._update_network()
                    train_iteration_metrics.append(metrics_dict)

                train_iteration_metrics = {k: [dic[k] for dic in train_iteration_metrics]
                                           for k in train_iteration_metrics[0]}

                for metric, value in train_iteration_metrics.items():
                    self.train_metrics.update(metric, np.mean(value))

                if self.use_sde:
                    self.train_metrics.update('sde_std', (self.actor.get_std()).mean().item())

                # soft update
                if cycle % self.target_update_interval == 0:
                    polyak_update(self.value_net.parameters(), self.target_net.parameters(), 0.005)

                success_rate = np.mean(cycle_summary_data['done'])
                reward = np.mean(cycle_summary_data['reward'])

                self.train_metrics.update('success_rate', success_rate)
                self.train_metrics.update('reward', reward)
                self.log_models_parameters()

                print("Epoch: {:4d}, Step: {:4d}, reward: {:3.2f}, success_rate: {:3.2f}".format(epoch, cycle,
                                                                                                 reward,
                                                                                                 success_rate))

            success_rate, reward, images = self._eval_agent(epoch)

            self.val_metrics.update('success_rate', success_rate)
            self.val_metrics.update('reward', reward)

            if self.should_save_episode_video and epoch % self.episode_video_timer == 0:
                imageio.mimsave(self.video_log_path.format(epoch), images)

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
        self.train_metrics.reset()
        native_observation = self.env.reset()

        observation = native_observation['observation']
        achieved_goal = native_observation['achieved_goal']
        desired_goal = native_observation['desired_goal']

        if self.use_sde:
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

    def _update_normalizer(self, observation, achieved_goal, desired_goal, action):
        # get the number of normalization transitions
        num_transitions = action.shape[0]
        # create the new buffer to store them
        t0, t1, t2, reward_batch, done_batch = self.her_module.sample_her_transitions(observation, achieved_goal,
                                                                                      desired_goal,
                                                                                      action,
                                                                                      num_transitions)

        (observation_batch, _, desired_goal_batch, actions) = t0

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

    def warmup(self):
        for step in range(1):
            cycle_summary_data = {'done': [], 'reward': []}
            cycle_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [], 'action': []}

            for _ in range(self.n_warmap_episodes):
                observation, achieved_goal, desired_goal, done, reward = self._reset()

                episode_summary_data = {'done': [], 'reward': []}
                episode_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [], 'action': []}

                for episode_step in range(self._max_episode_steps):
                    input_tensor = self._preprocess_inputs(observation, desired_goal)
                    action = self._select_action(input_tensor)

                    episode_data['observation'].append(observation.copy())
                    episode_data['achieved_goal'].append(achieved_goal.copy())
                    episode_data['desired_goal'].append(desired_goal.copy())
                    episode_data['action'].append(action.copy())

                    # feed the actions into the environment
                    new_observation, reward, _, info = self.env.step(action)

                    episode_summary_data['reward'].append(np.mean(reward))
                    episode_summary_data['done'].append(info['is_success'])

                    observation = new_observation['observation']
                    achieved_goal = new_observation['achieved_goal']

                cycle_data['observation'].append(np.asarray(episode_data['observation'], dtype=np.float32))
                cycle_data['achieved_goal'].append(np.asarray(episode_data['achieved_goal'], dtype=np.float32))
                cycle_data['desired_goal'].append(np.asarray(episode_data['desired_goal'], dtype=np.float32))
                cycle_data['action'].append(np.asarray(episode_data['action'], dtype=np.float32))

                cycle_summary_data['done'].append(np.mean(episode_summary_data['done']))
                cycle_summary_data['reward'].append(np.mean(episode_summary_data['reward']))

            cycle_data['observation'] = np.asarray(cycle_data['observation'], dtype=np.float32)
            cycle_data['achieved_goal'] = np.asarray(cycle_data['achieved_goal'], dtype=np.float32)
            cycle_data['desired_goal'] = np.asarray(cycle_data['desired_goal'], dtype=np.float32)
            cycle_data['action'] = np.asarray(cycle_data['action'], dtype=np.float32)

            cycle_summary_data['done'] = np.asarray(cycle_summary_data['done'], dtype=np.float32)
            cycle_summary_data['reward'] = np.asarray(cycle_summary_data['reward'], dtype=np.float32)

            # store the episodes
            self.buffer.store_episode(**cycle_data, n_episodes_to_store=self.n_warmap_episodes)
            self._update_normalizer(**cycle_data)


def make_env(config):
    if config.render_mode == 'none':
        env = gym.make(config.env_id)
    else:
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
    agent.train()


if __name__ == '__main__':
    train_agent_according_config(get_config(env_id='NeedleReach-v0', device='cpu'))
