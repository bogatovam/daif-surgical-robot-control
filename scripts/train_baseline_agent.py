import pkgutil
from typing import Tuple, Optional

import surrol.gym as surrol_gym

import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from config.configs_reader import config
from model.daif_pmdp_agent import Agent
from utils import set_random_seed
import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import should_collect_more_steps

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

ENV_ID = 'NeedleReach-v0'


def make_env(env_id):
    env = gym.make(env_id)
    env = Monitor(env, config.env_log_folder)
    env.seed(config.seed)
    return env


def main():
    env = make_env(ENV_ID)
    print(env)

    print(f'Actions count: {env.action_space.shape}')
    print(f'Actions size: {env.action_size}')
    # a = env.env_method('get_oracle_action', [env.get_original_obs()])
    print(f'Observation: {float(env.action_space.high[0])}')
    #
    # # Select action randomly or according to policy
    # actions, buffer_actions = sample_action(learning_starts, obs, action_noise, env.num_envs)
    #
    # # Rescale and perform action
    # new_obs, rewards, dones, infos = env.step(actions)


if __name__ == '__main__':
    main()
