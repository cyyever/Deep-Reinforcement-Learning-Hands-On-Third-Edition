#!/usr/bin/env python3
import argparse
import collections
import typing as tt

import gymnasium as gym
import numpy as np
import torch
from lib import dqn_model, wrappers

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", required=True, help="Directory for video")
    args = parser.parse_args()

    env = wrappers.make_env(args.env, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c: dict[int, int] = collections.Counter()

    while True:
        state_v = torch.tensor(np.expand_dims(state, 0))
        q_vals = net(state_v).data.numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        if is_done or is_trunc:
            break
    print(f"Total reward: {total_reward:.2f}")
    print("Action counts:", c)
    env.close()

