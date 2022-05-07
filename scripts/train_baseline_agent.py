import datetime
import os.path
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import surrol.gym as surrol_gym
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from base import BaseModel
from config.configs_reader import get_config
import gym

from config.const import ROOT_DIR_PATH
from logger import TensorboardWriter
from utils import MetricTracker

import threading
import numpy as np


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


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

        (_, achieved_goal_batch_t2, desired_goal_batch_t2, _) = t2
        # Recompute the reward for the augmented 'desired_goal'
        reward_batch = self.reward_func(achieved_goal_batch_t2, desired_goal_batch_t2, info=None)
        # Recompute the termination state for the augmented 'desired_goal'
        done_batch = self.done_func(achieved_goal_batch_t2, desired_goal_batch_t2, info=None)

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
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        # buf /= MPI.COMM_WORLD.Get_size()
        buf /= 1
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)


class MLP(BaseModel):
    def __init__(self, input_size, layer_sizes, output_size, lr=1e-3, output_activation=torch.nn.Identity,
                 activation=torch.nn.ELU, device='cpu'):
        super(MLP, self).__init__()
        sizes = [input_size] + layer_sizes + [output_size]

        self.layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            self.layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]

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

        self.transition_net = MLP(self.state_size + self.action_dim, [64, 64], self.state_size, lr=1e-3,
                                  device=self.device)

        self.prediction_policy_mu_network = MLP(self.state_size, [64] * 2, self.action_dim, device=self.device)
        self.prediction_policy_logstd_network = MLP(self.state_size, [64] * 2, self.action_dim, device=self.device)

        self.value_net = MLP(self.n_sampled_actions * (self.state_size + self.action_dim),
                             [1024] + [512] + [256] + [64] * 2,
                             self.n_sampled_actions, device=self.device)
        self.target_net = MLP(self.n_sampled_actions * (self.state_size + self.action_dim),
                              [1024] + [512] + [256] + [64] * 2,
                              self.n_sampled_actions, device=self.device)

        self.her_module = HERSampler(config.replay_strategy, config.replay_k, self.env.compute_reward,
                                     self.env.is_success)
        # create the replay buffer
        self.buffer = ReplayBuffer(self.env, config.max_episode_steps, self.memory_capacity,
                                   self.her_module.sample_her_transitions, config.device_id)

        self.o_norm = Normalizer(size=env.observation_space.spaces['observation'].shape[0])
        self.g_norm = Normalizer(size=env.observation_space.spaces['desired_goal'].shape[0])

        self.writer = TensorboardWriter(prepare_path(config.tb_log_folder, experiment_name=config.experiment_name),
                                        True)

        self.metrics = MetricTracker('vfe', 'efe_mse_loss', 'success_rate', 'reward', writer=self.writer)

    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        t0, t1, t2, reward_batch, done_batch = self.buffer.sample(self.batch_size)

        # Retrieve a batch for 3 consecutive points in time
        (observation_batch_t0, achieved_goal_batch_t0, desired_goal_batch_t0, actions_batch_t0) = t0
        (observation_batch_t1, achieved_goal_batch_t1, desired_goal_batch_t1, actions_batch_t1) = t1
        (observation_batch_t2, achieved_goal_batch_t2, desired_goal_batch_t2, actions_batch_t2) = t2

        state_batch_t0 = torch.cat((self.as_tensor(observation_batch_t0), self.as_tensor(desired_goal_batch_t0)), dim=1)
        state_batch_t1 = torch.cat((self.as_tensor(observation_batch_t1), self.as_tensor(desired_goal_batch_t1)), dim=1)
        state_batch_t2 = torch.cat((self.as_tensor(observation_batch_t2), self.as_tensor(desired_goal_batch_t2)), dim=1)

        actions_batch_t0, actions_batch_t1 = self.as_tensor(actions_batch_t0), self.as_tensor(actions_batch_t1)
        reward_batch, done_batch = self.as_tensor(reward_batch), self.as_tensor(done_batch)

        # At time t0 predict the state at time t1:
        # append actions vector nearby state
        X = torch.cat((state_batch_t0, actions_batch_t0), dim=1)
        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
            pred_batch_t0t1, state_batch_t1, reduction='none'), dim=1).unsqueeze(1)

        return (state_batch_t0, state_batch_t1, state_batch_t2, actions_batch_t0,
                actions_batch_t1, reward_batch, done_batch, pred_error_batch_t0t1)

    def compute_value_net_loss(self, obs_batch_t1, obs_batch_t2,
                               action_batch_t1, reward_batch_t1,
                               done_batch_t2, pred_error_batch_t0t1):

        with torch.no_grad():
            # Determine the action distribution for time t2:
            mu_t2, std_t2 = self.prediction(obs_batch_t2)

            distribution = torch.distributions.normal.Normal(loc=mu_t2, scale=torch.exp(std_t2))
            sampled_actions_batch_t2 = distribution.sample(sample_shape=[self.n_sampled_actions])

            # Determine the target EFEs for time t2:
            repeated_obs_batch = obs_batch_t2.unsqueeze(1).repeat(1, self.n_sampled_actions, 1)
            targe_net_input = torch.cat([repeated_obs_batch, sampled_actions_batch_t2.transpose(0, 1)], dim=2)
            targe_net_input = torch.flatten(targe_net_input, start_dim=1)
            target_expected_free_energies_batch_t2 = self.target_net(targe_net_input)

            # todo better way?
            probs = torch.exp(distribution.log_prob(sampled_actions_batch_t2).transpose(0, 1).sum(2))
            # Weigh the target EFEs according to the action distribution:
            weighted_targets = (probs * target_expected_free_energies_batch_t2).sum(-1).unsqueeze(1)

            # Determine the batch of bootstrapped estimates of the EFEs:
            expected_free_energy_estimate_batch = (
                    -reward_batch_t1 + pred_error_batch_t0t1 + self.beta * weighted_targets)

        mu_t1, std_t1 = self.prediction(obs_batch_t1)
        distribution = torch.distributions.normal.Normal(loc=mu_t1, scale=torch.exp(std_t1))
        sampled_actions_batch_t1 = distribution.sample(sample_shape=[self.n_sampled_actions])

        # Determine the Expected free energy at time t1 according to the value network:
        repeated_obs_batch = obs_batch_t1.unsqueeze(1).repeat(1, self.n_sampled_actions, 1)
        value_net_input = torch.cat([repeated_obs_batch, sampled_actions_batch_t1.transpose(0, 1)], dim=2)
        value_net_input = torch.flatten(value_net_input, start_dim=1)

        probs = torch.exp(distribution.log_prob(sampled_actions_batch_t1).transpose(0, 1).sum(2))

        efe_batch_t1 = (probs * self.value_net(value_net_input)).sum(-1).unsqueeze(1)

        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(expected_free_energy_estimate_batch, efe_batch_t1)

        return value_net_loss

    def compute_variational_free_energy(self, obs_batch_t1, action_batch_t1, pred_error_batch_t0t1):
        # Determine the action distribution for time t1:
        # policy_batch_t1 = self.policy_net(obs_batch_t1)

        mu_t1, std_t1 = self.prediction(obs_batch_t1)
        distribution = torch.distributions.normal.Normal(loc=mu_t1, scale=torch.exp(std_t1))
        sampled_actions_batch_t1 = distribution.sample(sample_shape=[self.n_sampled_actions])

        # Determine the Expected free energy at time t1 according to the value network:
        repeated_obs_batch = obs_batch_t1.unsqueeze(1).repeat(1, self.n_sampled_actions, 1)
        value_net_input = torch.cat([repeated_obs_batch, sampled_actions_batch_t1.transpose(0, 1)], dim=2)
        probs = torch.exp(distribution.log_prob(sampled_actions_batch_t1).transpose(0, 1).sum(2))
        value_net_input = torch.flatten(value_net_input, start_dim=1)

        expected_free_energy_t1 = self.value_net(value_net_input).detach()

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_expected_free_energy_batch_t1 = torch.softmax(-self.gamma * expected_free_energy_t1, dim=1).clamp(
            min=1e-9, max=1 - 1e-9)

        # Weigh them according to the action distribution:
        energy_batch = -(probs * torch.log(boltzmann_expected_free_energy_batch_t1)).sum(-1).view(
            self.batch_size, 1)

        # Determine the entropy of the action distribution
        entropy_batch = -(distribution.entropy()).sum(-1).view(self.batch_size, 1)

        # Determine the Variable Free Energy, then take the mean over all batch samples:
        vfe_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
        vfe = torch.mean(vfe_batch)

        return vfe

    def _update_network(self):
        # Retrieve transition data in mini batches:
        (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
         action_batch_t1, reward_batch_t1, done_batch_t2,
         pred_error_batch_t0t1) = self.get_mini_batches()

        # Compute the value network loss:
        value_net_loss = self.compute_value_net_loss(obs_batch_t1, obs_batch_t2,
                                                     action_batch_t1, reward_batch_t1,
                                                     done_batch_t2, pred_error_batch_t0t1)

        # Compute the variational free energy:
        vfe = self.compute_variational_free_energy(obs_batch_t1, action_batch_t1, pred_error_batch_t0t1)
        # Reset the gradients:
        self.transition_net.optimizer.zero_grad()
        self.prediction_policy_mu_network.optimizer.zero_grad()
        self.prediction_policy_logstd_network.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()

        # Compute the gradients:
        vfe.backward()
        value_net_loss.backward()

        # todo check this (from her)
        # actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # clip the q value
        # clip_return = 1 / (1 - self.args.gamma)
        # target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # Perform gradient descent:
        self.transition_net.optimizer.step()
        self.prediction_policy_mu_network.optimizer.step()
        self.prediction_policy_logstd_network.optimizer.step()
        self.value_net.optimizer.step()

        return vfe.item(), value_net_loss.item()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

    # do the evaluation
    def _eval_agent(self, epoch):
        images = []
        total_reward = []
        total_success_rate = []

        for cycle in range(10):
            per_reward = []
            per_success_rate = []

            observation, _, desired_goal, _, reward = self._reset()

            for _ in range(self._max_episode_steps):
                input_tensor = self._preprocess_inputs(observation, desired_goal)
                action = self._select_action(input_tensor)

                # feed the actions into the environment
                new_observation, reward, _, info = self.env.step(action)

                observation = new_observation['observation']
                per_reward.append(reward)
                per_success_rate.append(info['is_success'])

                if self.should_save_episode_video and epoch % self.episode_video_timer == 0:
                    images += [self.env.render()]

            total_reward.append(np.mean(per_reward))
            total_success_rate.append(np.mean(per_success_rate))

        total_reward = np.asarray(total_reward)
        total_success_rate = np.asarray(total_success_rate)
        return np.mean(total_success_rate), np.mean(total_reward), np.asarray(images)

    def train(self):
        print("Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now()))

        for epoch in range(self.n_epochs):
            for step in range(self.n_cycles):
                self.writer.set_step(self.n_cycles * epoch + step)

                cycle_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [], 'action': []}
                for _ in range(self.n_episodes):
                    observation, achieved_goal, desired_goal, done, reward = self._reset()

                    episode_data = {'observation': [], 'achieved_goal': [], 'desired_goal': [], 'action': []}

                    for _ in range(self._max_episode_steps):
                        input_tensor = self._preprocess_inputs(observation, desired_goal)
                        action = self._select_action(input_tensor)

                        episode_data['observation'].append(observation.copy())
                        episode_data['achieved_goal'].append(achieved_goal.copy())
                        episode_data['desired_goal'].append(desired_goal.copy())
                        episode_data['action'].append(action.copy())

                        # feed the actions into the environment
                        new_observation, _, _, _ = self.env.step(action)

                        observation = new_observation['observation']
                        achieved_goal = new_observation['achieved_goal']

                    cycle_data['observation'].append(np.asarray(episode_data['observation'], dtype=np.float32))
                    cycle_data['achieved_goal'].append(np.asarray(episode_data['achieved_goal'], dtype=np.float32))
                    cycle_data['desired_goal'].append(np.asarray(episode_data['desired_goal'], dtype=np.float32))
                    cycle_data['action'].append(np.asarray(episode_data['action'], dtype=np.float32))

                cycle_data['observation'] = np.asarray(cycle_data['observation'], dtype=np.float32)
                cycle_data['achieved_goal'] = np.asarray(cycle_data['achieved_goal'], dtype=np.float32)
                cycle_data['desired_goal'] = np.asarray(cycle_data['desired_goal'], dtype=np.float32)
                cycle_data['action'] = np.asarray(cycle_data['action'], dtype=np.float32)

                # store the episodes
                self.buffer.store_episode(**cycle_data, n_episodes_to_store=self.n_episodes)
                self._update_normalizer(**cycle_data)

                vfe = []
                value_net_loss = []
                for _ in range(self.n_batches):
                    # train the network
                    vfe_item, value_net_loss_item = self._update_network()
                    vfe.append(vfe_item), value_net_loss.append(value_net_loss_item)

                self.metrics.update('vfe', np.mean(vfe))
                self.metrics.update('efe_mse_loss', np.mean(value_net_loss))
                # soft update
                self._soft_update_target_network(self.target_net, self.value_net)
                val_success_rate, val_reward, images = self._eval_agent(epoch)
                self.log_models_parameters()

                print(
                    "Epoch: {:4d}, Step: {:4d}, reward: {:3.2f}, success_rate: {:3.2f}".format(epoch, step, val_reward,
                                                                                               val_success_rate))

                # plot_grad_flow(self.prediction_policy_mu_network.named_parameters())

            # start to do the evaluation
            val_success_rate, val_reward, images = self._eval_agent(epoch)

            self.metrics.update('success_rate', val_success_rate)
            self.metrics.update('reward', val_reward)

            # if self.should_save_episode_video and epoch % self.episode_video_timer == 0:
            #     self.writer.add_video(self.experiment_name + f'_{epoch}', images)

            if self.should_save_model and epoch > 0 and epoch % self.model_save_timer == 0:
                self.transition_net.save(os.path.join(self.model_path, 'transition_net.pth'))
                self.prediction_policy_mu_network.save(os.path.join(self.model_path, 'policy_mu_network.pth'))
                self.prediction_policy_logstd_network.save(os.path.join(self.model_path, 'policy_logstd_network.pth'))
                self.value_net.save(os.path.join(self.model_path, 'value_net.pth'))

        self.env.close()

        if self.should_save_model:
            self.transition_net.save(os.path.join(self.final_model_path, 'transition_net.pth'))
            self.prediction_policy_mu_network.save(os.path.join(self.final_model_path, 'policy_mu_network.pth'))
            self.prediction_policy_logstd_network.save(os.path.join(self.final_model_path, 'policy_logstd_network.pth'))
            self.value_net.save(os.path.join(self.final_model_path, 'value_net.pth'))

        # Print and keep a (.txt) record of stuff
        print("Training finished at {}".format(datetime.datetime.now()))

    def log_models_parameters(self):
        # add histogram of model parameters to the tensorboard
        for name, p in self.transition_net.named_parameters():
            self.writer.add_histogram('transition_net_net_' + name, p, bins='auto')
        for name, p in self.prediction_policy_mu_network.named_parameters():
            self.writer.add_histogram('prediction_policy_mu_network_net_' + name, p, bins='auto')
        for name, p in self.prediction_policy_logstd_network.named_parameters():
            self.writer.add_histogram('prediction_policy_logstd_network_net_' + name, p, bins='auto')
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
            # Determine the action distribution given the current observation:
            mu, std = self.prediction(input_tensor)
            distribution = torch.distributions.normal.Normal(loc=mu, scale=torch.exp(std))
            action = distribution.sample().squeeze(0).detach().cpu().numpy().flatten()

            # add the gaussian
            action += self.noise_eps * self.action_max * np.random.randn(*action.shape)
            action = np.clip(action, -self.action_max, self.action_max)
            # random actions...
            random_actions = np.random.uniform(low=-self.action_max, high=self.action_max, size=self.action_dim)
            # choose if use the random actions
            action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
            return action

    def prediction(self, input_tensor):
        mu = self.prediction_policy_mu_network(input_tensor)
        log_std = self.prediction_policy_logstd_network(input_tensor)
        # log_std = torch.clamp(log_std, *(-20, 2))
        return mu, log_std

    def _preprocess_inputs(self, observation, goal):
        observation, goal = self._preprocess_observation_and_goal(observation, goal)
        obs_norm = self.o_norm.normalize(observation)
        goal_norm = self.g_norm.normalize(goal)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, goal_norm])
        return torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _preprocess_observation_and_goal(self, o, g):
        o = np.clip(o, -200, 200)
        g = np.clip(g, -200, 200)
        return o, g

    def _update_normalizer(self, observation, achieved_goal, desired_goal, action):
        # get the number of normalization transitions
        num_transitions = action.shape[0]
        # create the new buffer to store them

        # todo need to use at least tree experiments
        t0, t1, t2, reward_batch, done_batch = self.her_module.sample_her_transitions(observation, achieved_goal,
                                                                                      desired_goal, action,
                                                                                      num_transitions)

        (observation_batch_t0, _, desired_goal_batch_t0, _) = t0

        observation_batch, desired_goal_batch = self._preprocess_observation_and_goal(observation_batch_t0,
                                                                                      desired_goal_batch_t0)
        # update
        self.o_norm.update(observation_batch)
        self.g_norm.update(desired_goal_batch)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

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

    if device == 'gpu':
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
