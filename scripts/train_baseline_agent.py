import surrol.gym as surrol_gym

from stable_baselines3.common.monitor import Monitor
from config.configs_reader import get_config
import gym


def make_env(config):
    env = gym.make(config.env_id)
    env = Monitor(env, config.env_log_folder)
    env.seed(config.seed)
    return env


def main(config):
    env = make_env(config)

    print(f'Actions count: {env.action_space.shape}')
    print(f'Action UB:   {float(env.action_space.high[0])}')
    print(f'Action LB: {float(env.action_space.low[0])}')


if __name__ == '__main__':
    main(get_config(env_id='NeedleReach-v0', device='cpu'))
