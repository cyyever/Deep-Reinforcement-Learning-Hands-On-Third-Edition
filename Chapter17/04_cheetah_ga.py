#!/usr/bin/env python3
import copy
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lib import common
from torch.utils.tensorboard.writer import SummaryWriter

NOISE_STD = 0.01
POPULATION_SIZE = 2000
PARENTS_COUNT = 10
WORKERS_COUNT = 6
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1


class Net(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hid_size: int = 64):
        super().__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(x)


def mutate_net(net: Net, seed: int, copy_net: bool = True) -> Net:
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        p.data += NOISE_STD * noise_t
    return new_net


def build_net(env: gym.Env, seeds: list[int]) -> Net:
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


@dataclass
class OutputItem:
    seeds: list[int]
    reward: float
    steps: int


def worker_func(input_queue: mp.Queue, output_queue: mp.Queue):
    env = gym.make("HalfCheetah-v4")
    cache = {}

    while True:
        parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1])
                else:
                    net = build_net(env, net_seeds)
            else:
                net = build_net(env, net_seeds)
            new_cache[net_seeds] = net
            reward, steps = common.evaluate(env, net, get_max_action=False)
            output_queue.put(OutputItem(seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-cheetah-ga")

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                s = list(population[parent][0]) + [next_seed]
                seeds.append(tuple(s))
            worker_queue.put(seeds)
        gen_idx += 1

    pass
