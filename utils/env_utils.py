import numpy as np


def get_env_parameters(env):
    # Get spaces parameters
    state_dim = 0
    state_dim += env.observation_space.spaces['observation'].shape[0]
    state_dim += env.observation_space.spaces['desired_goal'].shape[0]

    action_dim = env.action_space.shape[0]
    max_action_value = float(env.action_space.high[0])

    return state_dim, action_dim, max_action_value


def preprocess_obs(state):
    if isinstance(state, dict):
        state = np.concatenate((state['observation'], state['desired_goal']), -1)

    return state
