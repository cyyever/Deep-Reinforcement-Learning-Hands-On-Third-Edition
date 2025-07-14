#!/usr/bin/env python3
import gymnasium as gym
import time
import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard.writer import SummaryWriter

from lib import common


MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.001
LEARNING_RATE = 0.001



class Net(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_step(net: Net, batch_noise: list[common.TNoise], batch_reward: list[float],
               writer: SummaryWriter, step_idx: int):
    weighted_noise = None
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    m_updates = []
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)


if __name__ == "__main__":
    writer = SummaryWriter(comment="-cartpole-es")
    env = gym.make("CartPole-v1")

    net = Net(env.observation_space.shape[0], env.action_space.n)
    print(net)

    step_idx = 0
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            noise, neg_noise = common.sample_noise(net)
            batch_noise.append(noise)
            batch_noise.append(neg_noise)
            reward, steps = common.eval_with_noise(
                env, net, noise, NOISE_STD)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = common.eval_with_noise(
                env, net, neg_noise, NOISE_STD)
            batch_reward.append(reward)
            batch_steps += steps
            if batch_steps > MAX_BATCH_STEPS:
                break

        step_idx += 1
        m_reward = float(np.mean(batch_reward))
        if m_reward > 199:
            print("Solved in %d steps" % step_idx)
            break

        train_step(net, batch_noise, batch_reward, writer, step_idx)
        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        print("%d: reward=%.2f, speed=%.2f f/s" % (
            step_idx, m_reward, speed))
