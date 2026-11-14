"""Configuration for sparse terminal-reward LunarLander experiments."""

import json
import math
from pathlib import Path

from .exploration import validate_switching_config


def validate_config(config):
    expected = {
        "env_id", "seed", "device", "cpu_threads", "observation_input", "make_sparse",
        "max_episode_steps", "epochs", "steps_per_epoch", "hidden_sizes", "gamma",
        "epsilon", "replay_capacity", "batch_size", "replay_updates_per_episode",
        "learning_rate", "target_tau", "feature_dim", "q_scale", "error_epsilon",
        "max_grad_norm", "wall_minutes",
    }
    optional = {"checkpoint_every_epochs", "epsilon_switching",
                "epsilon_decay_epochs", "epsilon_decay_seconds"}
    missing = expected - set(config)
    unknown = set(config) - expected - optional
    if missing or unknown:
        raise ValueError(f"Config keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if config["env_id"] != "LunarLander-v3" or config["make_sparse"] is not True:
        raise ValueError("This experiment requires LunarLander-v3 and make_sparse=true")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config["observation_input"] not in ("raw", "raw_with_time"):
        raise ValueError("observation_input must be raw or raw_with_time")
    for name in ("cpu_threads", "max_episode_steps", "epochs", "steps_per_epoch",
                 "replay_capacity", "batch_size", "feature_dim"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seed", "replay_updates_per_episode"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("epsilon_decay_epochs", "epsilon_decay_seconds"):
        if config.get(name, 0) != 0:
            raise ValueError("Use constant epsilon or epsilon_switching; decay is not supported")
    checkpoint_every = config.get("checkpoint_every_epochs", 0)
    if type(checkpoint_every) is not int or checkpoint_every < 0:
        raise ValueError("checkpoint_every_epochs must be a nonnegative integer")
    sizes = config["hidden_sizes"]
    if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError("hidden_sizes must be a nonempty list of positive integers")
    for name in ("gamma", "epsilon", "learning_rate", "target_tau", "q_scale",
                 "error_epsilon", "max_grad_norm", "wall_minutes"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]):
            raise ValueError(f"{name} must be finite")
    if not 0 < config["gamma"] <= 1 or not 0 <= config["epsilon"] <= 1:
        raise ValueError("gamma must be in (0,1] and epsilon in [0,1]")
    for name in ("learning_rate", "q_scale", "error_epsilon", "max_grad_norm", "wall_minutes"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 < config["target_tau"] <= 1:
        raise ValueError("target_tau must be in (0,1]")
    if config["replay_capacity"] < config["max_episode_steps"]:
        raise ValueError("replay_capacity must fit a complete episode")
    if config["batch_size"] > config["replay_capacity"]:
        raise ValueError("batch_size cannot exceed replay_capacity")
    if config.get("epsilon_switching") is not None:
        validate_switching_config(config)
    return config


def load_config(path):
    with Path(path).open() as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    return validate_config(config)
