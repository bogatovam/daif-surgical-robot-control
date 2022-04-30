import time

import surrol.gym as surrol_gym

from stable_baselines3.common.monitor import Monitor
from config.configs_reader import get_config
import gym

from logger import Logger
from model.models import EnsembleModel, RewardModel
from training.agent import Agent
from training.buffer import Buffer
from training.normalizer import Normalizer
from training.planner import Planner
from training.rl_trainer import RlTrainer
from utils import utils, get_env_parameters


def make_env(config):
    env = gym.make(config.env_id, render_mode='human')
    env = Monitor(env, config.env_log_folder)
    env.seed(config.seed)
    return env


def main(config):
    env = make_env(config)
    utils.set_random_seed(config.seed, config.device_id)
    print(f'Actions count: {env.action_space.shape}')
    print(f'Action UB:   {float(env.action_space.high[0])}')
    print(f'Action LB: {float(env.action_space.low[0])}')

    state_dim, action_dim, max_action_value = get_env_parameters(env)

    normalizer = Normalizer()
    buffer = Buffer(state_dim, action_dim, config.ensemble_size, normalizer, device=config.device_id)

    ensemble = EnsembleModel(
        state_dim + action_dim,
        state_dim,
        config.hidden_size,
        config.ensemble_size,
        normalizer,
        device=config.device_id
    )
    reward_model = RewardModel(state_dim + action_dim, config.hidden_size, device=config.device_id)
    trainer = RlTrainer(
        ensemble,
        reward_model,
        buffer,
        n_train_epochs=config.n_train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        grad_clip_norm=config.grad_clip_norm
    )

    planner = Planner(
        ensemble,
        reward_model,
        action_dim,
        config.ensemble_size,
        plan_horizon=config.plan_horizon,
        optimisation_iters=config.optimisation_iters,
        n_candidates=config.n_candidates,
        top_candidates=config.top_candidates,
        use_reward=True,
        use_exploration=True,
        use_mean=True,
        expl_scale=0.1,
        reward_scale=1.0,
        device=config.device_id,
    )

    logger = Logger(config.log_folder, config.seed)
    agent = Agent(env, planner, logger=logger)

    agent.get_seed_episodes(buffer, config.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes | {} frames]"

    for episode in range(1, config.n_episodes):
        logger.log("\n=== Episode {} ===".format(episode))
        start_time = time.time()

        msg = "Training on [{}/{}] data points"
        logger.log(msg.format(buffer.total_steps, buffer.total_steps * 4))
        trainer.reset_models()
        ensemble_loss, reward_loss = trainer.train()
        logger.log_losses(ensemble_loss, reward_loss)

        recorder = None
        # recorder = VideoRecorder(env.unwrapped, path=filename)
        # logger.log("Setup recoder @ {}".format(filename))

        logger.log("\n=== Collecting data [{}] ===".format(episode))
        reward, steps, stats = agent.run_episode(
            buffer, recorder=recorder
        )
        logger.log_episode(reward, steps)
        logger.log_stats(stats)

        logger.log_time(time.time() - start_time)
        logger.save()


if __name__ == '__main__':
    main(get_config(env_id='NeedleReach-v0', device='cpu'))
