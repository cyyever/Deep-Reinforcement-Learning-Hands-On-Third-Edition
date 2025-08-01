#!/usr/bin/env python3
import argparse
import math
import os
import time

import gymnasium as gym
import numpy as np
import ptan
import torch
import torch.nn.functional as F
import torch.optim as optim
from lib import common, kfac, model
from torch.utils.tensorboard.writer import SummaryWriter

GAMMA = 0.99
REWARD_STEPS = 5
BATCH_SIZE = 32
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-3
ENVS_COUNT = 16

TEST_ITERS = 100000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", choices=list(common.ENV_PARAMS.keys()),
                        default='cheetah', help="Environment id, default=cheetah")
    parser.add_argument("--mujoco", default=False, action='store_true',
                        help="If given, MuJoCo env will be used instead of PyBullet")
    parser.add_argument("--no-unhealthy", default=False, action='store_true',
                        help="Disable unhealthy checks in MuJoCo env")
    args = parser.parse_args()
    device = torch.device(args.dev)
    name = args.name + "-mujoco" if args.mujoco else "-pybullet"
    save_path = os.path.join("saves", "acktr-" + name)
    os.makedirs(save_path, exist_ok=True)

    extra = {}
    if args.mujoco and args.no_unhealthy:
        extra['terminate_when_unhealthy'] = False

    env_id = common.register_env(args.env, args.mujoco)
    envs = [gym.make(env_id, **extra) for _ in range(ENVS_COUNT)]
    test_env = gym.make(env_id, **extra)

    net_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    net_crt = model.ModelCritic(envs[0].observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-acktr_" + name)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)

    opt_act = kfac.KFACOptimizer(net_act, lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps, strict=False)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = model.test_net(net_act, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(f"Best reward updated: {best_reward:.3f} -> {rewards:.3f}")
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net_act.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                mu_v = net_act(states_v)
                log_prob_v = model.calc_logprob(mu_v, net_act.logstd, actions_v)
                if opt_act.steps % opt_act.Ts == 0:
                    opt_act.zero_grad()
                    pg_fisher_loss = -log_prob_v.mean()
                    opt_act.acc_stats = True
                    pg_fisher_loss.backward(retain_graph=True)
                    opt_act.acc_stats = False

                opt_act.zero_grad()
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                loss_policy_v = -(adv_v * log_prob_v).mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * torch.exp(net_act.logstd)) + 1) / 2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
