"""Single-process Q-learning with values defined by normalized prediction error."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import time
from datetime import datetime, timezone

from .config import load_config


class ReplayBuffer:
    """Uniform transition replay, populated only by complete native episodes."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = []
        self.position = 0

    def extend(self, transitions):
        for transition in transitions:
            if len(self.transitions) < self.capacity:
                self.transitions.append(transition)
            else:
                self.transitions[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.transitions), size=min(batch_size, len(self.transitions)),
                             replace=False)
        return [self.transitions[int(index)] for index in indices]

    def __len__(self):
        return len(self.transitions)


def collect_episode(env, critic, config, rng, epsilon, reset_seed=None):
    """Finish the episode before admitting any of its transitions to learning."""
    import torch
    from .core import choose_action, encode_observation

    observation, _ = env.reset(seed=reset_seed)
    device = next(critic.parameters()).device
    transitions = []
    raw_return = 0.0
    sparse_return = 0.0
    horizon = config["max_episode_steps"]
    make_sparse = config["make_sparse"]
    for elapsed in range(horizon):
        state = encode_observation(observation, elapsed, horizon, config["observation_input"])
        with torch.no_grad():
            action = choose_action(critic, state.to(device), rng, epsilon)
        observation, reward, terminated, truncated, _ = env.step(action)
        stats_reward = float(reward)
        if not torch.isfinite(torch.tensor(stats_reward)):
            raise FloatingPointError("Nonfinite environment reward")
        bellman_reward = stats_reward
        if make_sparse:
            if not terminated:
                bellman_reward = 0.0
        if terminated and bellman_reward not in (-100.0, 100.0):
            raise ValueError(f"Unexpected native terminal reward: {bellman_reward}")
        next_state = encode_observation(observation, elapsed + 1, horizon,
                                        config["observation_input"])
        transitions.append((state, action, bellman_reward, next_state, bool(terminated)))
        raw_return += stats_reward
        sparse_return += bellman_reward
        if terminated or truncated:
            break
    metrics = {
        "steps": elapsed + 1, "environment_return": raw_return,
        "learning_return": sparse_return, "terminated": bool(terminated),
        "truncated": bool(truncated), "timeout": not terminated,
        "terminal_reward": stats_reward if terminated else None,
        "success": bool(terminated and stats_reward == 100.0),
        "failure": bool(terminated and stats_reward == -100.0),
    }
    return transitions if terminated else [], metrics


def make_batch(transitions, device):
    import torch
    return {
        "states": torch.stack([item[0] for item in transitions]).to(device),
        "actions": torch.tensor([item[1] for item in transitions], dtype=torch.long, device=device),
        "rewards": torch.tensor([item[2] for item in transitions], dtype=torch.float32, device=device),
        "next_states": torch.stack([item[3] for item in transitions]).to(device),
        "terminated": torch.tensor([item[4] for item in transitions], dtype=torch.bool, device=device),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_record(stream, record):
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()


def save_checkpoint(path, critic, optimizer, config, epoch, completed_epochs, total_steps, updates,
                    training_metadata=None):
    import torch
    checkpoint = {
        "method": "rnd_error_q", "format_version": 1, "config": config,
        "epoch": epoch, "completed_epochs": completed_epochs, "total_steps": total_steps,
        "updates": updates,
        "critic": {name: tensor.detach().cpu().clone() for name, tensor in critic.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "training_metadata": training_metadata,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def archive_checkpoint(source, destination, metadata):
    """Preserve an atomically published checkpoint without replacing an earlier snapshot."""
    import torch

    if destination.exists():
        raise FileExistsError(f"Checkpoint snapshot already exists: {destination}")
    temporary = destination.with_suffix(".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    content = destination.read_bytes()
    saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    actual_metadata = {**metadata, **(saved.get("training_metadata") or {}),
                       **{name: saved[name] for name in ("epoch", "completed_epochs", "total_steps", "updates")}}
    with (destination.parent / "checkpoint_index.jsonl").open("a") as stream:
        write_record(stream, {"checkpoint": destination.name,
                              "sha256": hashlib.sha256(content).hexdigest(), **actual_metadata})


def critic_digest(critic):
    digest = hashlib.sha256()
    for name, tensor in sorted(critic.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def monitor_policy(critic, env, config, deadline):
    """Measure a frozen greedy policy without touching training data or random streams."""
    from .evaluate import evaluate_episode, summarize

    settings = config["epsilon_switching"]
    before = critic_digest(critic)
    was_training = critic.training
    records = []
    started = time.perf_counter()
    critic.eval()
    try:
        for offset in range(settings["monitor_episodes"]):
            if time.perf_counter() >= deadline:
                break
            records.append(evaluate_episode(critic, env, config,
                                            settings["monitor_seed"] + offset))
    finally:
        critic.train(was_training)
    if critic_digest(critic) != before:
        raise RuntimeError("Greedy monitoring changed model tensors")
    return {"complete": len(records) == settings["monitor_episodes"],
            "model_sha256": before, "weights_unchanged": True,
            "seconds": time.perf_counter() - started,
            "steps": sum(record["steps"] for record in records),
            "summary": summarize(records), "episodes": records}


def restore_checkpoint(path, config, critic, optimizer, device):
    """Restore learned state while explicitly starting new replay and random streams."""
    import torch

    path = Path(path).resolve()
    content = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(content), map_location=device, weights_only=True)
    if checkpoint.get("method") != "rnd_error_q" or checkpoint.get("format_version") != 1:
        raise ValueError("Continuation requires an rnd_error_q format-1 checkpoint")
    allowed_changes = {"epsilon", "epsilon_decay_epochs", "epsilon_decay_seconds",
                       "seed", "epochs", "wall_minutes", "checkpoint_every_epochs"}
    saved_config = checkpoint["config"]
    switching = config.get("epsilon_switching") is not None
    if switching or saved_config.get("epsilon_switching") is not None:
        allowed_changes -= {"epsilon", "epsilon_decay_epochs", "epsilon_decay_seconds"}
    mismatches = sorted(name for name in set(config) | set(saved_config)
                        if name not in allowed_changes and config.get(name) != saved_config.get(name))
    if mismatches:
        raise ValueError(f"Continuation settings must match: {mismatches}")
    for name in ("completed_epochs", "total_steps", "updates"):
        if type(checkpoint[name]) is not int or checkpoint[name] < 0:
            raise ValueError(f"Invalid checkpoint counter: {name}")
    if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < -1:
        raise ValueError("Invalid checkpoint epoch")
    if config["epochs"] <= checkpoint["completed_epochs"]:
        raise ValueError("Continuation epoch limit must exceed completed epochs")
    controller_state = None
    if switching:
        from .exploration import ExplorationController
        controller = ExplorationController(config)
        controller_state = checkpoint.get("training_metadata", {}).get("exploration_controller")
        controller.load_state_dict(controller_state, checkpoint["completed_epochs"])
        if checkpoint["training_metadata"].get("epsilon") != controller.epsilon():
            raise ValueError("Checkpoint epsilon does not match its exploration controller")
    critic.load_state_dict(checkpoint["critic"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    metadata = {
        "checkpoint_path": path.name, "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"],
        "starting_completed_epochs": checkpoint["completed_epochs"],
        "starting_total_steps": checkpoint["total_steps"],
        "starting_total_updates": checkpoint["updates"],
        "epoch_restart": "Start at completed_epochs with a full new per-epoch step budget",
        "exploration_controller": controller_state,
    }
    return checkpoint, metadata


def train(config, output, continue_from=None):
    import gymnasium as gym
    import numpy as np
    import torch
    from .core import ErrorQNetwork
    from .updates import td_update

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select cpu explicitly")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    streams = np.random.SeedSequence(config["seed"]).spawn(3)
    reset_seed = int(streams[0].generate_state(1)[0])
    action_rng, replay_rng = (np.random.default_rng(stream) for stream in streams[1:])
    device = torch.device(config["device"])
    env = gym.make(config["env_id"], max_episode_steps=config["max_episode_steps"])
    monitoring_env = None
    started = time.perf_counter()
    try:
        if env.observation_space.shape != (8,) or env.action_space.n != 4:
            raise ValueError("Expected eight raw observations and four discrete actions")
        obs_dim = 9 if config["observation_input"] == "raw_with_time" else 8
        critic = ErrorQNetwork(obs_dim, 4, config["hidden_sizes"], config["feature_dim"],
                               config["q_scale"], config["error_epsilon"]).to(device)
        optimizer = torch.optim.Adam(critic.predictor.parameters(), lr=config["learning_rate"])
        continuation = None
        initial_epoch = -1
        total_steps = total_updates = completed_epochs = 0
        if continue_from is not None:
            checkpoint, continuation = restore_checkpoint(
                continue_from, config, critic, optimizer, device)
            initial_epoch = checkpoint["epoch"]
            total_steps = checkpoint["total_steps"]
            total_updates = checkpoint["updates"]
            completed_epochs = checkpoint["completed_epochs"]
        starting_steps, starting_updates, start_epoch = total_steps, total_updates, completed_epochs
        replay = ReplayBuffer(config["replay_capacity"])
        switching = config.get("epsilon_switching")
        controller = None
        if switching is not None:
            from .exploration import ExplorationController
            controller = ExplorationController(config)
            if continuation is not None:
                controller.load_state_dict(continuation["exploration_controller"], completed_epochs)
            monitoring_env = gym.make(config["env_id"], max_episode_steps=config["max_episode_steps"])
        monitoring_steps = monitoring_episodes = monitoring_checks = 0
        monitoring_seconds = 0.0
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", {
            "method": "rnd_error_q", "fresh_start": continue_from is None, "workers": 1,
            "continuation": continuation,
            "replay_reset": continue_from is not None,
            "random_streams_reset": continue_from is not None,
            "counter_scope": {
                "total_steps": "cumulative", "updates": "cumulative",
                "completed_epochs": "cumulative", "episodes": "segment",
                "outcomes": "segment", "epoch_steps": "segment",
                "monitoring": "segment", "exploration_controller": "cumulative",
            },
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "torch": torch.__version__,
            "gymnasium": gym.__version__, "numpy": np.__version__,
            "device": str(device), "environment_first_reset_seed": reset_seed,
            "trainable_parameters": sum(p.numel() for p in critic.predictor.parameters()),
            "total_parameters": sum(p.numel() for p in critic.parameters()),
            "wall_minutes": config["wall_minutes"],
            "checkpoint_every_epochs": config.get("checkpoint_every_epochs", 0),
            "exploration_schedule": {
                "initial_epsilon": config["epsilon"],
                "switching": switching,
                "time_origin": "Start of this segment's training loop",
                "update_boundary": "Episode start; epsilon stays fixed within each episode",
            },
        })
        total_episodes = 0
        accepted_updates = total_successes = total_failures = total_timeouts = 0
        initial_metadata = {
            "training_seconds": 0.0, "episode": 0,
            "epsilon": controller.epsilon() if controller else config["epsilon"],
            "exploration_controller": controller.state_dict() if controller else None,
            "monitoring_steps": 0, "monitoring_episodes": 0,
            "monitoring_checks": 0, "monitoring_seconds": 0.0,
        }
        save_checkpoint(output / "initial_checkpoint.pt", critic, optimizer, config, initial_epoch,
                        completed_epochs, total_steps, total_updates, initial_metadata)
        save_checkpoint(output / "checkpoint.pt", critic, optimizer, config, initial_epoch,
                        completed_epochs, total_steps, total_updates, initial_metadata)
        training_started = time.perf_counter()
        budget_started = training_started if controller else started
        epoch = start_epoch

        def current_epsilon(elapsed):
            return controller.epsilon() if controller else config["epsilon"]

        def checkpoint_metadata():
            elapsed = time.perf_counter() - training_started
            return {
                "training_seconds": elapsed, "episode": total_episodes,
                "epsilon": current_epsilon(elapsed),
                "exploration_controller": controller.state_dict() if controller else None,
                "monitoring_steps": monitoring_steps, "monitoring_episodes": monitoring_episodes,
                "monitoring_checks": monitoring_checks, "monitoring_seconds": monitoring_seconds,
            }

        def run_monitor():
            nonlocal monitoring_steps, monitoring_episodes, monitoring_checks, monitoring_seconds
            result = monitor_policy(critic, monitoring_env, config,
                                    budget_started + 60.0 * config["wall_minutes"])
            monitoring_steps += result["steps"]
            monitoring_episodes += len(result["episodes"])
            monitoring_checks += 1
            monitoring_seconds += result["seconds"]
            decision = (controller.observe(completed_epochs, result["summary"]["success_fraction"])
                        if result["complete"] else None)
            record = {"check": monitoring_checks, "completed_epochs": completed_epochs,
                      "total_training_steps": total_steps, "total_updates": total_updates,
                      "elapsed_training_seconds": time.perf_counter() - training_started,
                      "total_monitoring_steps": monitoring_steps,
                      "controller_decision": decision, **result}
            with (output / "monitoring.jsonl").open("a") as stream:
                write_record(stream, record)
            print(json.dumps({"event": "greedy_monitor", **{k: v for k, v in record.items()
                                                           if k != "episodes"}}), flush=True)

        published_metadata = {"epoch": initial_epoch, "completed_epochs": completed_epochs,
                              "total_steps": total_steps, "updates": total_updates,
                              **checkpoint_metadata()}
        with (output / "episodes.jsonl").open("x") as episode_log, \
             (output / "updates.jsonl").open("x") as update_log, \
             (output / "metrics.jsonl").open("x") as epoch_log:
            failure = None
            budget_reached = False
            try:
                if controller and controller.last_observed_epoch is None:
                    run_monitor()
                    next_metadata = {"epoch": initial_epoch, "completed_epochs": completed_epochs,
                                     "total_steps": total_steps, "updates": total_updates,
                                     **checkpoint_metadata()}
                    save_checkpoint(output / "checkpoint.pt", critic, optimizer, config, initial_epoch,
                                    completed_epochs, total_steps, total_updates, next_metadata)
                    published_metadata = next_metadata
                    budget_reached = time.perf_counter() - budget_started >= 60.0 * config["wall_minutes"]
                for epoch in range(start_epoch, config["epochs"]):
                    if budget_reached:
                        break
                    epoch_start = time.perf_counter()
                    epoch_mode = controller.state_dict()["mode"] if controller else None
                    epoch_steps = epoch_episodes = landings = crashes = timeouts = eligible_steps = 0
                    environment_returns = learning_returns = 0.0
                    updates_before, accepted_before = total_updates, accepted_updates
                    epsilon_start = None
                    while epoch_steps < config["steps_per_epoch"]:
                        epsilon_elapsed_seconds = time.perf_counter() - training_started
                        epsilon = current_epsilon(epsilon_elapsed_seconds)
                        if epsilon_start is None:
                            epsilon_start = epsilon
                        transitions, episode = collect_episode(
                            env, critic, config, action_rng, epsilon, reset_seed)
                        reset_seed = None
                        total_steps += episode["steps"]
                        epoch_steps += episode["steps"]
                        total_episodes += 1
                        epoch_episodes += 1
                        landings += int(episode["success"])
                        crashes += int(episode["failure"])
                        timeouts += int(episode["timeout"])
                        total_successes += int(episode["success"])
                        total_failures += int(episode["failure"])
                        total_timeouts += int(episode["timeout"])
                        eligible_steps += len(transitions)
                        environment_returns += episode["environment_return"]
                        learning_returns += episode["learning_return"]
                        replay.extend(transitions)
                        write_record(episode_log, {"episode": total_episodes, "epoch": epoch,
                                                  "total_steps": total_steps, "epsilon": epsilon,
                                                  "epsilon_elapsed_seconds": epsilon_elapsed_seconds,
                                                  "exploration_mode": epoch_mode,
                                                  "eligible_steps": len(transitions), **episode})

                        # Reverse chronological minibatches preserve the terminal-first replay order.
                        batches = [("reverse", transitions[max(0, end - config["batch_size"]):end][::-1])
                                   for end in range(len(transitions), 0, -config["batch_size"])]
                        if len(replay):
                            batches.extend(("replay", replay.sample(config["batch_size"], replay_rng))
                                           for _ in range(config["replay_updates_per_episode"]))
                        for phase, items in batches:
                            diagnostic = td_update(critic, optimizer, make_batch(items, device), config)
                            total_updates += 1
                            accepted_updates += int(diagnostic["accepted"])
                            write_record(update_log, {"update": total_updates, "epoch": epoch,
                                                      "episode": total_episodes, "phase": phase,
                                                      "samples": len(items), **diagnostic})
                        next_metadata = {"epoch": epoch, "completed_epochs": completed_epochs,
                                         "total_steps": total_steps, "updates": total_updates,
                                         **checkpoint_metadata()}
                        save_checkpoint(output / "checkpoint.pt", critic, optimizer, config, epoch,
                                        completed_epochs, total_steps, total_updates, next_metadata)
                        published_metadata = next_metadata
                        if total_episodes % 10 == 0:
                            print(json.dumps({"epoch": epoch, "total_steps": total_steps,
                                              "landings": total_successes, "crashes": total_failures,
                                              "timeouts": total_timeouts, "updates": total_updates,
                                              "epsilon": epsilon}), flush=True)
                        if time.perf_counter() - budget_started >= 60.0 * config["wall_minutes"]:
                            budget_reached = True
                            break
                    epoch_completed = epoch_steps >= config["steps_per_epoch"]
                    if epoch_completed:
                        completed_epochs = epoch + 1
                    if (controller and epoch_completed and not budget_reached
                            and completed_epochs % switching["monitor_every_epochs"] == 0):
                        run_monitor()
                        budget_reached = time.perf_counter() - budget_started >= 60.0 * config["wall_minutes"]
                    metrics = {
                        "epoch": epoch, "completed_epochs": completed_epochs,
                        "epoch_completed": epoch_completed,
                        "steps": epoch_steps, "total_steps": total_steps, "episodes": epoch_episodes,
                        "successes": landings, "failures": crashes, "timeouts": timeouts,
                        "success_fraction": landings / epoch_episodes, "eligible_steps": eligible_steps,
                        "excluded_timeout_steps": epoch_steps - eligible_steps,
                        "mean_environment_return": environment_returns / epoch_episodes,
                        "mean_learning_return": learning_returns / epoch_episodes,
                        "epsilon": epsilon, "epsilon_start": epsilon_start, "epsilon_end": epsilon,
                        "exploration_mode": epoch_mode,
                        "next_epsilon": current_epsilon(time.perf_counter() - training_started),
                        "exploration_controller": controller.state_dict() if controller else None,
                        "monitoring_steps": monitoring_steps, "monitoring_episodes": monitoring_episodes,
                        "elapsed_training_seconds": time.perf_counter() - training_started,
                        "updates": total_updates - updates_before,
                        "accepted_updates": accepted_updates - accepted_before,
                        "replay_size": len(replay), "seconds": time.perf_counter() - epoch_start,
                    }
                    write_record(epoch_log, metrics)
                    print(json.dumps(metrics, sort_keys=True), flush=True)
                    next_metadata = {"epoch": epoch, "completed_epochs": completed_epochs,
                                     "total_steps": total_steps, "updates": total_updates,
                                     **checkpoint_metadata()}
                    save_checkpoint(output / "checkpoint.pt", critic, optimizer, config, epoch,
                                    completed_epochs, total_steps, total_updates, next_metadata)
                    published_metadata = next_metadata
                    checkpoint_every = config.get("checkpoint_every_epochs", 0)
                    if epoch_completed and checkpoint_every and completed_epochs % checkpoint_every == 0:
                        archive_checkpoint(output / "checkpoint.pt", output / f"epoch_{completed_epochs:06d}.pt",
                                           {"kind": "epoch", **published_metadata})
                    if budget_reached:
                        break
                status = "time_budget" if budget_reached else "completed"
            except KeyboardInterrupt:
                # Keep the last atomically published checkpoint, not a partially interrupted update.
                status = "interrupted"
            except Exception as error:
                status, failure = "failed", error
        training_seconds = time.perf_counter() - training_started
        archive_checkpoint(output / "checkpoint.pt", output / "final_checkpoint.pt",
                           {"kind": "final", "stop_status": status, **published_metadata})
        write_json(output / "completion.json", {
            "status": status, "completed_epochs": completed_epochs, "total_steps": total_steps,
            "total_updates": total_updates, "segment_steps": total_steps - starting_steps,
            "counter_scope": "Episodes, outcomes, and attempted/accepted updates describe this segment",
            "episodes": total_episodes, "updates_attempted": total_updates - starting_updates,
            "updates_accepted": accepted_updates, "successes": total_successes,
            "failures": total_failures, "timeouts": total_timeouts,
            "seconds": time.perf_counter() - started,
            "training_seconds": training_seconds,
            "scheduled_epsilon_at_end": current_epsilon(training_seconds),
            "exploration_controller": controller.state_dict() if controller else None,
            "monitoring_steps": monitoring_steps, "monitoring_episodes": monitoring_episodes,
            "monitoring_checks": monitoring_checks, "monitoring_seconds": monitoring_seconds,
            "final_checkpoint": "final_checkpoint.pt",
            "error": f"{type(failure).__name__}: {failure}" if failure is not None else None,
        })
        if failure is not None:
            raise failure
    finally:
        if monitoring_env is not None:
            monitoring_env.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--continue-from", type=Path,
                        help="Restore a checkpoint into a new output directory; replay and RNG streams restart")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if args.output is None:
        parser.error("--execute requires a new --output directory")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    train(config, args.output, args.continue_from)


if __name__ == "__main__":
    main()
