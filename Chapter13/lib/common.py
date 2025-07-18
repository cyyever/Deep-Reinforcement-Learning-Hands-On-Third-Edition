import re
import typing as tt
import warnings
from datetime import datetime, timedelta
from types import SimpleNamespace

import gymnasium as gym
import ptan
import ptan.ignite as ptan_ignite
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from textworld import text_utils

PARAMS = {
    'small': SimpleNamespace(**{
        'encoder_size': 20,
        'embeddings': 20,
        'replay_size': 10000,
        'replay_initial': 1000,
        'sync_nets': 100,
        'epsilon_steps': 1000,
        'epsilon_final': 0.2,
    }),

    'medium': SimpleNamespace(**{
        'encoder_size': 256,
        'embeddings': 128,
        'replay_size': 100000,
        'replay_initial': 10000,
        'sync_nets': 200,
        'epsilon_steps': 10000,
        'epsilon_final': 0.2,
    })
}


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


def setup_ignite(engine: Engine, exp_source, run_name: str,
                 extra_metrics: tt.Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    handler = ptan_ignite.EndOfEpisodeHandler(exp_source)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        avg_steps = trainer.state.metrics.get('avg_steps', 50)
        avg_reward = trainer.state.metrics.get('avg_reward', 0.0)
        print("Episode %d: reward=%.0f (avg %.2f), "
              "steps=%s (avg %.2f), speed=%.1f f/s, "
              "elapsed=%s" % (
            trainer.state.episode,
            trainer.state.episode_reward, avg_reward,
            trainer.state.episode_steps, avg_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

        if avg_steps < 15 and trainer.state.episode > 100:
            print("Average steps has fallen below 15, stop training")
            trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward', 'avg_steps']
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


def get_games_spaces(game_files: list[str]) -> tuple[
    dict[int, str],
    gym.Space,
    gym.Space,
]:
    """
    Get games vocabulary, action and observation spaces
    :param game_files: game files to wrap
    :return: tuple with dictionary, action and observation spaces
    """
    vocab = text_utils.extract_vocab_from_gamefiles(game_files)
    vocab_dict = {
        idx: word
        for idx, word in enumerate(sorted(vocab))
    }
    word_space = gym.spaces.Discrete(len(vocab))
    action_space = gym.spaces.Sequence(word_space)
    observation_space = gym.spaces.Sequence(word_space)
    return vocab_dict, action_space, observation_space


def build_rev_vocab(vocab: dict[int, str]) -> dict[str, int]:
    """
    Build reverse vocabulary
    :param vocab: forward vocab (int -> word)
    :return: reverse vocabulary (word -> int)
    """
    res = {word: idx for idx, word in vocab.items()}
    assert len(res) == len(vocab)
    return res


def tokenize(text: str, rev_vocab: dict[str, int]) -> list[int]:
    """
    Very simple tokeniser into fixed word set
    :param text: text to tokenize
    :param rev_vocab: reverse vocabulary
    :return: list of tokens
    """
    res = []
    for word in re.split(r'\W+', text.lower()):
        token = rev_vocab.get(word)
        if token is not None:
            res.append(token)
    return res