#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import ptan
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from lib import common
from ptan.common.utils import TBMeanTracker
from ptan.experience import VectorExperienceSourceFirstLast
from torch.utils.tensorboard.writer import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 8

GRAD_BATCH = 64
TRAIN_BATCH = 2


ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18


def make_env() -> gym.Env:
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


def grads_func(proc_name: str, net: common.AtariA2C, device: torch.device,
               train_queue: mp.Queue):
    env_factories = [make_env for _ in range(NUM_ENVS)]
    env = gym.vector.SyncVectorEnv(env_factories)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = VectorExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    batch = []
    frame_idx = 0
    writer = SummaryWriter(comment=proc_name)

    with common.RewardTracker(writer, REWARD_BOUND) as tracker:
        with TBMeanTracker(writer, 100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards and tracker.reward(new_rewards[0], frame_idx):
                    break

                batch.append(exp)
                if len(batch) < GRAD_BATCH:
                    continue

                data = common.unpack_batch(batch, net, device=device, gamma=GAMMA,
                                           reward_steps=REWARD_STEPS)
                states_v, actions_t, vals_ref_v = data
                batch.clear()

                net.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_p_a = log_prob_v[range(GRAD_BATCH), actions_t]
                log_prob_actions_v = adv_v * log_p_a
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                ent = (prob_v * log_prob_v).sum(dim=1).mean()
                entropy_loss_v = ENTROPY_BETA * ent

                loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                loss_v.backward()

                tb_tracker.track("advantage", adv_v, frame_idx)
                tb_tracker.track("values", value_v, frame_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, frame_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v, frame_idx)
                tb_tracker.track("loss_value", loss_value_v, frame_idx)
                tb_tracker.track("loss_total", loss_v, frame_idx)

                # gather gradients
                nn_utils.clip_grad_norm_(
                    net.parameters(), CLIP_GRAD)
                grads = [
                    param.grad.data.cpu().numpy() if param.grad is not None else None
                    for param in net.parameters()
                ]
                train_queue.put(grads)

    train_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        proc_name = f"-a3c-grad_pong_{args.name}#{proc_idx}"
        p_args = (proc_name, net, device, train_queue)
        data_proc = mp.Process(target=grads_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry, strict=False):
                    tgt_grad += grad

            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer, strict=False):
                    param.grad = torch.FloatTensor(grad).to(device)

                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
