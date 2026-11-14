"""Frozen greedy evaluation and visualization of a saved neural Q-function."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .config import validate_config
from .core import ErrorQNetwork, choose_action, encode_observation


def load_checkpoint(path):
    contents = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("method") != "rnd_error_q"
            or type(checkpoint.get("format_version")) is not int
            or checkpoint["format_version"] != 1):
        raise ValueError("Expected an rnd_error_q checkpoint (format 1)")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Checkpoint config must be an object")
    config = validate_config(checkpoint["config"])
    obs_dim = 9 if config["observation_input"] == "raw_with_time" else 8
    critic = ErrorQNetwork(obs_dim, 4, config["hidden_sizes"], config["feature_dim"],
                           q_scale=config["q_scale"], error_epsilon=config["error_epsilon"])
    critic.load_state_dict(checkpoint["critic"])
    critic.eval()
    critic.requires_grad_(False)
    metadata = {"checkpoint_sha256": hashlib.sha256(contents).hexdigest(),
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_completed_epochs": checkpoint["completed_epochs"],
                "checkpoint_total_steps": checkpoint["total_steps"]}
    return critic, config, metadata


def viewer_closed():
    import pygame

    if not pygame.display.get_init():
        return True
    return any(event.type == pygame.QUIT
               or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
               for event in pygame.event.get([pygame.QUIT, pygame.KEYDOWN]))


@torch.no_grad()
def evaluate_episode(critic, env, config, seed, render=False):
    observation, _ = env.reset(seed=seed)
    initial_observation = observation.tolist()
    rng = np.random.default_rng(seed)
    raw_return = 0.0
    terminal_reward = None
    horizon = config["max_episode_steps"]
    for step in range(1, horizon + 1):
        if render and viewer_closed():
            return None
        state = encode_observation(observation, step - 1, horizon,
                                   observation_input=config["observation_input"])
        action = choose_action(critic, state, rng, epsilon=0.0)
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        if not np.isfinite(reward):
            raise FloatingPointError("Nonfinite environment reward")
        raw_return += reward
        if terminated:
            terminal_reward = reward
            if terminal_reward not in (-100.0, 100.0):
                raise ValueError(f"Unexpected native terminal reward: {terminal_reward}")
            break
        if truncated:
            break
    return {"seed": seed, "initial_observation": initial_observation, "steps": step, "raw_return": raw_return,
            "sparse_return": terminal_reward if terminated else 0.0,
            "terminal_reward": terminal_reward, "terminated": bool(terminated),
            "truncated": bool(truncated), "timeout": not bool(terminated),
            "censored": not bool(terminated),
            "success": terminal_reward == 100.0 if terminated else None}


def summarize(records):
    count = len(records)
    successes = sum(record["success"] is True for record in records)
    crashes = sum(record["success"] is False for record in records)
    return {"episodes": count, "successes": successes, "crashes": crashes,
            "timeouts": count - successes - crashes, "censored": count - successes - crashes,
            "success_fraction": successes / count if count else None,
            "mean_raw_return": sum(record["raw_return"] for record in records) / count if count else None,
            "mean_sparse_return": sum(record["sparse_return"] for record in records) / count if count else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3000001],
                        help="Environment seeds, one independent environment per seed")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per seed; zero means indefinite rendered viewing")
    parser.add_argument("--seed-mode", choices=("stream", "episode"), default="stream",
                        help="Seed once per stream, or increment the seed for every episode")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 0 or any(seed < 0 for seed in args.seeds):
        parser.error("Episodes and seeds must be nonnegative")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Evaluation seeds must be distinct")
    if not args.episodes and (not args.render or len(args.seeds) != 1 or args.output):
        parser.error("Indefinite viewing requires --render, one seed, and no --output")
    if args.output and args.output.exists():
        parser.error("Evaluation output already exists")

    torch.set_num_threads(1)
    critic, config, metadata = load_checkpoint(args.checkpoint)
    records, streams = [], []
    interrupted = False
    try:
        for stream_seed in args.seeds:
            env = gym.make(config["env_id"], max_episode_steps=config["max_episode_steps"],
                           render_mode="human" if args.render else None)
            stream_records = []
            completed = 0
            try:
                while not args.episodes or completed < args.episodes:
                    if args.seed_mode == "episode":
                        reset_seed = stream_seed + completed
                    else:
                        reset_seed = stream_seed if completed == 0 else None
                    record = evaluate_episode(critic, env, config, reset_seed, args.render)
                    if record is None:
                        interrupted = True
                        break
                    record.update(stream_seed=stream_seed, episode=completed + 1)
                    if args.episodes:
                        stream_records.append(record)
                    completed += 1
                    print(json.dumps(record, sort_keys=True), flush=True)
            except KeyboardInterrupt:
                interrupted = True
            finally:
                env.close()
            if args.episodes:
                records.extend(stream_records)
                streams.append({"seed": stream_seed, "summary": summarize(stream_records)})
            if interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True

    if args.episodes:
        result = {"method": "rnd_error_q", "action_selection": "greedy", "device": "cpu",
                  **metadata, "seed_mode": args.seed_mode, "seeds": args.seeds,
                  "requested_episodes_per_seed": args.episodes, "interrupted": interrupted,
                  "observation_input": config["observation_input"],
                  "max_episode_steps": config["max_episode_steps"],
                  "summary": summarize(records), "streams": streams, "episodes": records}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
