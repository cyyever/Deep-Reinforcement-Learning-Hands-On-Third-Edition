#!/usr/bin/env python3
import argparse
import random
import warnings
from datetime import datetime, timedelta

import gymnasium as gym
import ptan
import ptan.ignite as ptan_ignite
import torch
import torch.optim as optim
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from lib import common, dqn_model

NAME = "02_n_envs"


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int, steps: int):
    buffer.populate(initial)
    while True:
        buffer.populate(steps)
        yield buffer.sample(batch_size)


if __name__ == "__main__":
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.GAME_PARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("--envs", type=int, default=3, help="Amount of environments to run in parallel")
    args = parser.parse_args()
    device = torch.device(args.dev)

    envs = [
        ptan.common.wrappers.wrap_dqn(gym.make(params.env_name))
        for _ in range(args.envs)
    ]
    params.batch_size *= args.envs

    net = dqn_model.DQN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=params.gamma, env_seed=common.SEED)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration * args.envs)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(exp_source, bound_avg_reward=18.0).attach(engine)
    ptan_ignite.EpisodeFPSHandler(fps_mul=args.envs).attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=trainer.state.metrics.get('time_passed', 0))))
        trainer.should_terminate = trainer.state.episode > 700

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics['time_passed']),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{params.run_name}-{NAME}={args.envs}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss', 'avg_fps'],
                                      output_transform=lambda a: a)
    tb.attach(engine, log_handler=handler, event_name=ptan_ignite.PeriodEvents.ITERS_100_COMPLETED)

    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size, args.envs))
