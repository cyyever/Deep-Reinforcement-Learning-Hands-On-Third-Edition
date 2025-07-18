#!/usr/bin/env python3
import argparse

import gym
import ptan
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from lib import common, dqn_model
from tensorboardX import SummaryWriter

PLAY_STEPS = 4


def play_func(params, net, cuda, exp_queue):
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    device = torch.device("cuda" if cuda else "cpu")

    writer = SummaryWriter(comment="-" + params.run_name + "-03_parallel")

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma, steps_count=1)
    exp_source_iter = iter(exp_source)

    frame_idx = 0

    with common.RewardTracker(writer, params.stop_reward) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

    exp_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    params = common.HYPERPARAMS['pong']
    params.batch_size *= PLAY_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue))
    play_proc.start()

    frame_idx = 0

    while play_proc.is_alive():
#        frame_idx += PLAY_STEPS
        # for _ in range(PLAY_STEPS):
        while exp_queue.qsize() > 1:
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)
            frame_idx += 1
            if frame_idx % params.target_net_sync == 0:
                tgt_net.sync()

        if len(buffer) < params.replay_initial:
            continue
        optimizer.zero_grad()
        batch = buffer.sample(params.batch_size)
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()

