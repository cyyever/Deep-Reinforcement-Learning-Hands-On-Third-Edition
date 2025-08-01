import collections
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import ptan
import ptan.ignite as ptan_ignite
import torch
import torch.nn as nn
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.engine import Engine
from ignite.metrics import RunningAverage

SEED = 123


def unpack_batch(batch: list[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.asarray(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.asarray(exp.last_state)
        last_states.append(lstate)
    return np.asarray(states, dtype=np.float32), \
           np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=bool), \
           np.asarray(last_states, dtype=np.float32)


def calc_loss_dqn(
        batch: list[ptan.experience.ExperienceFirstLast],
        net: nn.Module,
        tgt_net: nn.Module,
        gamma: float,
        device: torch.device = torch.device("cpu")
):
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)


def calc_loss_double_dqn(
        batch: list[ptan.experience.ExperienceFirstLast],
        net: nn.Module,
        tgt_net: nn.Module,
        gamma: float,
        device: torch.device = torch.device("cpu")
):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_acts = net(next_states_v).max(1)[1]
        next_state_acts = next_state_acts.unsqueeze(-1)
        next_state_vals = tgt_net(next_states_v).gather(
                1, next_state_acts).squeeze(-1)
        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, exp_sa_vals)


class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - \
              frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


@torch.no_grad()
def calc_values_of_states(states: np.ndarray, net: nn.Module, device: torch.device):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    warnings.simplefilter("ignore", category=UserWarning)
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.2f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)


class PseudoCountRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, hash_function=lambda o: o,
                 reward_scale: float = 1.0):
        super().__init__(env)
        self.hash_function = hash_function
        self.reward_scale = reward_scale
        self.counts = collections.Counter()

    def step(self, action):
        obs, reward, done, is_tr, info = self.env.step(action)
        extra_reward = self._count_observation(obs)
        return obs, reward + self.reward_scale * extra_reward, done, is_tr, info

    def _count_observation(self, obs) -> float:
        """
        Increments observation counter and returns pseudo-count reward
        :param obs: observation
        :return: extra reward
        """
        h = self.hash_function(obs)
        self.counts[h] += 1
        return np.sqrt(1 / self.counts[h])


class NetworkDistillationRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_callable, reward_scale: float = 1.0, sum_rewards: bool = True):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_callable = reward_callable
        self.sum_rewards = sum_rewards

    def step(self, action):
        obs, reward, done, is_tr, info = self.env.step(action)
        extra_reward = self.reward_callable(obs)
        if self.sum_rewards:
            res_rewards = reward + self.reward_scale * extra_reward
        else:
            res_rewards = np.array([reward, extra_reward * self.reward_scale])
        return obs, res_rewards, done, is_tr, info


class DistillExperienceSource(ptan.experience.ExperienceSource):
    """
    Tweaked version of experience source which sums up reward
    """
    def pop_rewards_steps(self):
        res = []
        for rewards, steps in super().pop_rewards_steps():
            res.append((rewards.sum(), steps))
        return res
