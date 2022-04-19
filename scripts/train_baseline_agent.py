import pkgutil
import surrol.gym as surrol_gym

import gym

from stable_baselines3.common.env_util import make_vec_env

ENV_ID = 'NeedleReach-v0'


def main():
    env = gym.make(ENV_ID)
    print("hehheeh")


if __name__ == '__main__':
    main()
