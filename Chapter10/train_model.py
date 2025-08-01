#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import ptan
import torch
import torch.optim as optim
from gymnasium import wrappers
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.engine import Engine
from lib import common, data, environ, models, validation

SAVES_DIR = pathlib.Path("saves")
STOCKS = "data/YNDX_160101_161231.csv"
VAL_STOCKS = "data/YNDX_150101_151231.csv"

BATCH_SIZE = 32
BARS_COUNT = 10

EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", help="Training device name", default="cpu")
    parser.add_argument("--data", default=STOCKS, help=f"Stocks file or dir, default={STOCKS}")
    parser.add_argument("--year", type=int, help="Year to train on, overrides --data")
    parser.add_argument("--val", default=VAL_STOCKS, help="Validation data, default=" + VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device(args.dev)

    saves_path = SAVES_DIR / f"simple-{args.run}"
    saves_path.mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path(args.data)
    val_path = pathlib.Path(args.val)

    if args.year is not None or data_path.is_file():
        if args.year is not None:
            stock_data = data.load_year_data(args.year)
        else:
            stock_data = {"YNDX": data.load_relative(data_path)}
        env = environ.StocksEnv(
            stock_data, bars_count=BARS_COUNT)
        env_tst = environ.StocksEnv(
            stock_data, bars_count=BARS_COUNT)
    elif data_path.is_dir():
        env = environ.StocksEnv.from_dir(
            data_path, bars_count=BARS_COUNT)
        env_tst = environ.StocksEnv.from_dir(
            data_path, bars_count=BARS_COUNT)
    else:
        raise RuntimeError("No data to train on")

    env = wrappers.TimeLimit(env, max_episode_steps=1000)
    val_data = {"YNDX": data.load_relative(val_path)}
    env_val = environ.StocksEnv(val_data, bars_count=BARS_COUNT)

    net = models.SimpleFFDQN(env.observation_space.shape[0],
                             env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START)
    eps_tracker = ptan.actions.EpsilonTracker(
        selector, EPS_START, EPS_FINAL, EPS_STEPS)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss(
            batch, net, tgt_net.target_model,
            gamma=GAMMA ** REWARD_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        eps_tracker.frame(engine.state.iteration)

        if getattr(engine.state, "eval_states", None) is None:
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.asarray(transition.state)
                           for transition in eval_states]
            engine.state.eval_states = np.asarray(eval_states)

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    tb = common.setup_ignite(engine, exp_source, f"simple-{args.run}",
                             extra_metrics=('values_mean',))

    @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def sync_eval(engine: Engine):
        tgt_net.sync()

        mean_val = common.calc_values_of_states(
            engine.state.eval_states, net, device=device)
        engine.state.metrics["values_mean"] = mean_val
        if getattr(engine.state, "best_mean_val", None) is None:
            engine.state.best_mean_val = mean_val
        if engine.state.best_mean_val < mean_val:
            print("%d: Best mean value updated %.3f -> %.3f" % (
                engine.state.iteration, engine.state.best_mean_val,
                mean_val))
            path = saves_path / (f"mean_value-{mean_val:.3f}.data")
            torch.save(net.state_dict(), path)
            engine.state.best_mean_val = mean_val

    @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def validate(engine: Engine):
        res = validation.validation_run(env_tst, net, device=device)
        print("%d: tst: %s" % (engine.state.iteration, res))
        for key, val in res.items():
            engine.state.metrics[key + "_tst"] = val
        res = validation.validation_run(env_val, net, device=device)
        print("%d: val: %s" % (engine.state.iteration, res))
        for key, val in res.items():
            engine.state.metrics[key + "_val"] = val
        val_reward = res['episode_reward']
        if getattr(engine.state, "best_val_reward", None) is None:
            engine.state.best_val_reward = val_reward
        if engine.state.best_val_reward < val_reward:
            print(f"Best validation reward updated: {engine.state.best_val_reward:.3f} -> {val_reward:.3f}, model saved")
            engine.state.best_val_reward = val_reward
            path = saves_path / (f"val_reward-{val_reward:.3f}.data")
            torch.save(net.state_dict(), path)

    event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED
    tst_metrics = [m + "_tst" for m in validation.METRICS]
    tst_handler = tb_logger.OutputHandler(
        tag="test", metric_names=tst_metrics)
    tb.attach(engine, log_handler=tst_handler, event_name=event)

    val_metrics = [m + "_val" for m in validation.METRICS]
    val_handler = tb_logger.OutputHandler(
        tag="validation", metric_names=val_metrics)
    tb.attach(engine, log_handler=val_handler, event_name=event)

    engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))
