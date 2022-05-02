import datetime
import os.path
import random
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.monitor import Monitor

from base import BaseModel
from config.configs_reader import get_config
import gym

from config.const import ROOT_DIR_PATH
from logger import TensorboardWriter
from utils import MetricTracker


class ReplayMemory:
    def __init__(self, capacity, obs_shape, actions_shape, device='cpu'):
        self.device = device

        self.capacity = capacity  # The maximum number of items to be stored in memory

        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([capacity] + [dim for dim in obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty([capacity] + [dim for dim in actions_shape], dtype=torch.float32,
                                      device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.float32, device=self.device)
        self.done_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.push_count = 0  # The number of times new data has been pushed to memory

    def push(self, obs, action, reward, done):
        # Store data to memory
        self.obs_mem[self.position()] = obs
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.done_mem[self.position()] = done

        self.push_count += 1

    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity

    def sample(self, obs_indices, action_indices, reward_indices, done_indices, max_n_indices, batch_size):
        # Fine as long as max_n is not greater than the fewest number of time steps an episode can take

        # Pick indices at random
        end_indices = np.random.choice(min(self.push_count, self.capacity) - max_n_indices * 2, batch_size,
                                       replace=False) + max_n_indices

        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position() + max_n_indices):
                end_indices[i] += max_n_indices

        # Retrieve the specified indices that come before the end_indices
        obs_batch = self.obs_mem[np.array([index - obs_indices for index in end_indices])]
        action_batch = self.action_mem[np.array([index - action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index - reward_indices for index in end_indices])]
        done_batch = self.done_mem[np.array([index - done_indices for index in end_indices])]

        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.done_mem[index - j]:
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0])
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[
                                                                      0])  # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(
                                self.reward_mem[0])  # Reward of 0 will probably not make sense for every environment
                    for k in range(len(done_indices)):
                        if done_indices[k] >= j:
                            done_batch[i, k] = torch.zeros_like(self.done_mem[0])
                    break

        return obs_batch, action_batch, reward_batch, done_batch


class MLP(BaseModel):
    def __init__(self, input_size, layer_sizes, output_size, lr=1e-3, output_activation=torch.nn.Identity,
                 activation=torch.nn.ELU, device='cpu'):
        super(MLP, self).__init__()
        sizes = [input_size] + layer_sizes + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
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
        self.state_dim, self.action_dim, self.max_action_value = get_env_parameters(env)

        self.device = config.device_id

        self.n_episodes = int(config.n_episodes)
        self.batch_size = int(config.batch_size)
        self.memory_capacity = int(config.memory_capacity)

        self.freeze_period = int(config.freeze_period)

        self.gamma = float(config.gamma)  # A precision parameter
        self.beta = float(config.beta)  # The discount rate
        self.print_timer = int(config.print_timer)
        self.should_save_model = interpret_boolean(config.should_save_model)
        self.model_path = prepare_path(config.model_path, config.experiment_name)
        self.final_model_path = os.path.join(self.model_path, "final")
        self.model_save_timer = int(config.network_save_timer)

        self.state_shape = np.add(self.env.observation_space['observation'].shape,
                                  self.env.observation_space['desired_goal'].shape)
        self.state_size = np.prod(self.state_shape)

        self.actions_shape = self.env.action_space.shape
        self.action_dim = self.env.action_space.shape[-1]

        self.n_sampled_actions = config.n_sampled_actions
        self.freeze_cntr = 0

        self.transition_net = MLP(self.state_size + self.action_dim, [64, 64], self.state_size, lr=1e-3,
                                  device=self.device)

        self.prediction_policy_mu_network = MLP(self.state_size, [64] * 2, self.action_dim)
        self.prediction_policy_logstd_network = MLP(self.state_size, [64] * 2, self.action_dim)

        self.value_net = MLP(self.state_size + self.action_dim, [64] * 2, 1)
        self.target_net = MLP(self.state_size + self.action_dim, [64] * 2, 1)
        self.memory = ReplayMemory(self.memory_capacity, self.state_shape, self.actions_shape, device=self.device)

        # When sampling from memory at index i, obs_indices indicates that we want observations
        # with indices i-obs_indices, works the same for the others
        self.obs_indices = [2, 1, 0]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1

        self.writer = TensorboardWriter(config.log_dir, True)

        self.metrics = MetricTracker('vfe', 'efe_mse_loss', 'success_rate', 'reward', writer=self.writer)

    def prediction(self, encoded_state):
        mu = self.prediction_policy_mu_network(encoded_state)
        log_std = self.prediction_policy_logstd_network(encoded_state)
        log_std = torch.clamp(log_std, *(-20, 2))
        return mu, log_std

    def select_action(self, obs):
        with torch.no_grad():
            # Determine the action distribution given the current observation:
            mu, std = self.prediction(obs)
            distribution = torch.distributions.normal.Normal(mu, std)
            return distribution.sample().squeeze(0).detach().cpu().numpy().flatten()

    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.memory.sample(
            self.obs_indices, self.action_indices, self.reward_indices,
            self.done_indices, self.max_n_indices, self.batch_size)

        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.state_shape])
        obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.state_shape])
        obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.state_shape])

        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0]
        action_batch_t1 = all_actions_batch[:, 1]

        # At time t0 predict the state at time t1:
        # append actions vector nearby state
        X = torch.cat((obs_batch_t0, action_batch_t0), dim=1)
        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
            pred_batch_t0t1, obs_batch_t1, reduction='none'), dim=1).unsqueeze(1)

        return (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
                action_batch_t1, reward_batch_t1, done_batch_t2, pred_error_batch_t0t1)

    def compute_value_net_loss(self, obs_batch_t1, obs_batch_t2,
                               action_batch_t1, reward_batch_t1,
                               done_batch_t2, pred_error_batch_t0t1):

        with torch.no_grad():
            # Determine the action distribution for time t2:
            mu, std = self.prediction(obs_batch_t2)

            distribution = torch.distributions.normal.Normal(mu, std)
            sampled_actions_batch_t1 = distribution.sample(sample_shape=self.n_sampled_actions)
            probs = distribution.log_prob(sampled_actions_batch_t1)

            # Determine the target EFEs for time t2:
            target_expected_free_energies_batch_t2 = self.target_net(
                torch.cat((obs_batch_t1, sampled_actions_batch_t1), dim=1))

            # Weigh the target EFEs according to the action distribution:
            weighted_targets = ((1 - done_batch_t2) * probs *
                                target_expected_free_energies_batch_t2).sum(-1).unsqueeze(1)

            # Determine the batch of bootstrapped estimates of the EFEs:
            expected_free_energy_estimate_batch = (
                    -reward_batch_t1 + pred_error_batch_t0t1 + self.beta * weighted_targets)

        # Determine the Expected free energy at time t1 according to the value network:
        efe_batch_t1 = self.value_net(torch.cat((obs_batch_t1, action_batch_t1), dim=1))

        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(expected_free_energy_estimate_batch, efe_batch_t1)

        return value_net_loss

    def compute_variational_free_energy(self, obs_batch_t1, pred_error_batch_t0t1):
        # Determine the action distribution for time t1:
        # policy_batch_t1 = self.policy_net(obs_batch_t1)

        mu, std = self.prediction(obs_batch_t1)  # (batch_size, 5)

        distribution = torch.distributions.normal.Normal(mu, std)

        # Determine the Expected free energys for time t1:
        expected_free_energy_t1 = self.value_net(obs_batch_t1).detach()

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_expected_free_energy_batch_t1 = torch.softmax(-self.gamma * expected_free_energy_t1, dim=1).clamp(
            min=1e-9, max=1 - 1e-9)

        sampled_actions_batch_t1 = distribution.sample(sample_shape=self.n_sampled_actions)
        probs = distribution.log_prob(sampled_actions_batch_t1)
        # Weigh them according to the action distribution:
        energy_batch = -(probs * torch.log(boltzmann_expected_free_energy_batch_t1)).sum(-1).view(
            self.batch_size, 1)

        # Determine the entropy of the action distribution
        entropy_batch = -(distribution.entropy()).sum(-1).view(self.batch_size, 1)

        # Determine the Variable Free Energy, then take the mean over all batch samples:
        vfe_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
        vfe = torch.mean(vfe_batch)

        return vfe

    def learn(self):
        # If there are not enough transitions stored in memory, return:
        if self.memory.push_count - self.max_n_indices * 2 < self.batch_size:
            return

        # After every freeze_period time steps, update the target network:
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1

        # Retrieve transition data in mini batches:
        (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
         action_batch_t1, reward_batch_t1, done_batch_t2,
         pred_error_batch_t0t1) = self.get_mini_batches()

        # Compute the value network loss:
        value_net_loss = self.compute_value_net_loss(obs_batch_t1, obs_batch_t2,
                                                     action_batch_t1, reward_batch_t1,
                                                     done_batch_t2, pred_error_batch_t0t1)

        # Compute the variational free energy:
        vfe = self.compute_variational_free_energy(obs_batch_t1, pred_error_batch_t0t1)
        # Reset the gradients:
        self.transition_net.optimizer.zero_grad()
        self.prediction_policy_mu_network.optimizer.zero_grad()
        self.prediction_policy_logstd_network.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()

        # Compute the gradients:
        vfe.backward()
        value_net_loss.backward()

        # Perform gradient descent:
        self.transition_net.optimizer.step()
        self.prediction_policy_mu_network.optimizer.step()
        self.prediction_policy_logstd_network.optimizer.step()
        self.value_net.optimizer.step()

        self.metrics.update('vfe', vfe.item())
        self.metrics.update('efe_mse_loss', value_net_loss.item())

    def train(self):
        print("Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now()))

        results = {'reward': [], 'done': []}
        for ith_episode in range(self.n_episodes):
            self.writer.set_step(ith_episode)

            total_reward = 0
            self.metrics.reset()
            obs = self.env.reset()
            obs = preprocess_obs(obs)
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            done = False
            reward = 0
            while not done:
                action = self.select_action(obs)
                self.memory.push(obs, action, reward, done)

                obs, reward, done, _ = self.env.step(action.cpu().data.numpy())

                done = bool(done)
                obs = preprocess_obs(obs)
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                total_reward += reward

                results['reward'] += [reward]
                results['done'] += [done]

                self.learn()
                if done:
                    self.memory.push(obs, -99, -99, done)

            # Print and keep a (.txt) record of stuff
            avg_reward = np.mean(results['reward'])
            success_rate = np.mean(results['done'])

            self.metrics.update('success_rate', success_rate)
            self.metrics.update('reward', avg_reward)

            if ith_episode > 0 and ith_episode % self.print_timer == 0:
                last_x = np.mean(results[-self.print_timer:])

                print("Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward,
                                                                                            self.print_timer, last_x))
            if self.should_save_model and ith_episode > 0 and ith_episode % self.model_save_timer == 0:
                self.transition_net.save(os.path.join(self.model_path, 'transition_net.pth'))
                self.prediction_policy_mu_network.save(os.path.join(self.model_path, 'policy_mu_network.pth'))
                self.prediction_policy_logstd_network.save(os.path.join(self.model_path, 'policy_logstd_network.pth'))
                self.value_net.save(os.path.join(self.model_path, 'value_net.pth'))

            self.log_models_parameters()

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
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.prediction_policy_mu_network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.prediction_policy_logstd_network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.value_net.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')


def preprocess_obs(state):
    if isinstance(state, dict):
        state = np.concatenate((state['observation'], state['desired_goal']), -1)

    return state


def make_env(config):
    env = gym.make(config.env_id, render_mode=config.render_mode)
    env = Monitor(env, prepare_path(config.env_log_folder, config.experiment_name))
    env.seed(config.seed)
    return env


def interpret_boolean(param):
    if type(param) == bool:
        return param
    elif param in ['True', '1']:
        return True
    elif param in ['False', '0']:
        return False
    else:
        sys.exit("param '{}' cannot be interpreted as boolean".format(param))


def get_env_parameters(env):
    # Get spaces parameters
    state_dim = 0
    state_dim += env.observation_space.spaces['observation'].shape[0]
    state_dim += env.observation_space.spaces['desired_goal'].shape[0]

    action_dim = env.action_space.shape[0]
    max_action_value = float(env.action_space.high[0])

    return state_dim, action_dim, max_action_value


def prepare_path(path, unique_id):
    return os.path.join(ROOT_DIR_PATH, unique_id, path)


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

    # todo tensorboard writer
    # todo noise
    # todo her
    # todo gif


if __name__ == '__main__':
    train_agent_according_config(get_config(env_id='NeedleReach-v0', device='cpu'))
