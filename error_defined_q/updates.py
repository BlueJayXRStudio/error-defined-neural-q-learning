"""One-step TD updates for values defined by random-target prediction errors."""

import math

import torch

from .core import update_target


@torch.no_grad()
def td_targets(critic, batch, gamma):
    """Detached targets; native terminal samples never evaluate a next value."""
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one")
    parameter = next(critic.predictor.parameters())
    rewards = torch.as_tensor(batch["rewards"], dtype=parameter.dtype, device=parameter.device)
    terminated = torch.as_tensor(batch["terminated"], dtype=torch.bool, device=parameter.device)
    next_states = torch.as_tensor(batch["next_states"], dtype=parameter.dtype, device=parameter.device)
    if (rewards.ndim != 1 or terminated.shape != rewards.shape
            or next_states.shape != (len(rewards), critic.obs_dim)):
        raise ValueError("Inconsistent Bellman target batch shapes")
    result = rewards.clone()
    continuing = ~terminated
    if gamma and continuing.any():
        result[continuing] += gamma * critic.q_values(next_states[continuing], target=True).max(dim=-1).values
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite Bellman targets")
    return result


def td_update(critic, optimizer, batch, config):
    """Fit a detached Bellman target, then update the lagged predictor."""
    tau = float(config.get("target_tau", 0.005))
    max_grad_norm = float(config.get("max_grad_norm", 10.0))
    if (not math.isfinite(tau) or not 0 < tau <= 1
            or not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("Invalid target rate or gradient norm limit")
    parameter = next(critic.predictor.parameters())
    states = torch.as_tensor(batch["states"], dtype=parameter.dtype, device=parameter.device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=parameter.device)
    if (states.ndim != 2 or not len(states) or states.shape[1] != critic.obs_dim
            or actions.shape != (len(states),) or not torch.isfinite(states).all()):
        raise ValueError("Expected a nonempty finite batch of states and actions")
    targets = td_targets(critic, batch, float(config.get("gamma", 0.99)))
    if targets.shape != actions.shape:
        raise ValueError("Target and action batch sizes must match")
    error, initial_error = critic.errors(states, actions)
    predictions = critic.values_from_errors(error, initial_error)
    residual = (predictions - targets) / critic.q_scale
    loss = 0.5 * residual.square().mean()
    if not torch.isfinite(predictions).all() or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite predictions or TD loss")
    with torch.no_grad():
        prediction64 = predictions.double()
        target64 = targets.double()
        error_ratio = (error.double() + critic.error_epsilon) / (initial_error.double() + critic.error_epsilon)
        report = dict(
            loss_before=float(loss.double().item()),
            prediction_mean=float(prediction64.mean().item()),
            prediction_min=float(prediction64.min().item()),
            prediction_max=float(prediction64.max().item()),
            target_mean=float(target64.mean().item()),
            target_min=float(target64.min().item()),
            target_max=float(target64.max().item()),
            td_abs_mean=float((prediction64 - target64).abs().mean().item()),
            error_ratio_mean=float(error_ratio.mean().item()),
        )
    if not all(math.isfinite(value) for value in report.values()):
        raise FloatingPointError("Nonfinite TD diagnostics")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        critic.predictor.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    report["grad_norm"] = float(grad_norm.double().item())
    if not math.isfinite(report["grad_norm"]):
        raise FloatingPointError("Nonfinite predictor gradient norm")
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in critic.predictor.parameters()):
        raise FloatingPointError("Nonfinite predictor parameters after optimizer step")
    update_target(critic, tau)
    report["accepted"] = True
    return report
