"""Signed action values defined by changes in random-target prediction error."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def feature_network(input_dim, hidden_sizes, feature_dim):
    layers = []
    width = input_dim
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    layers.append(nn.Linear(width, feature_dim))
    return nn.Sequential(*layers)


class ErrorQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, feature_dim,
                 q_scale=100.0, error_epsilon=1e-8):
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        if any(type(size) is not int or size <= 0
               for size in (obs_dim, act_dim, feature_dim, *hidden_sizes)):
            raise ValueError("Network dimensions must be positive integers")
        if (not math.isfinite(q_scale) or q_scale <= 0
                or not math.isfinite(error_epsilon) or error_epsilon <= 0):
            raise ValueError("Q scale and error epsilon must be finite and positive")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.q_scale = float(q_scale)
        self.error_epsilon = float(error_epsilon)
        self.random_target = feature_network(obs_dim + act_dim, hidden_sizes, feature_dim)
        self.predictor = feature_network(obs_dim + act_dim, hidden_sizes, feature_dim)
        self.initial_predictor = copy.deepcopy(self.predictor)
        self.target_predictor = copy.deepcopy(self.predictor)
        for network in (self.random_target, self.initial_predictor, self.target_predictor):
            network.requires_grad_(False)

    def action_inputs(self, states, actions):
        parameter = next(self.predictor.parameters())
        states = torch.as_tensor(states, dtype=parameter.dtype, device=parameter.device)
        unbatched = states.ndim == 1
        if unbatched:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[1] != self.obs_dim:
            raise ValueError("Expected one observation or a batch of observation vectors")
        actions = torch.as_tensor(actions, dtype=torch.long, device=parameter.device)
        if unbatched and actions.ndim == 0:
            actions = actions.unsqueeze(0)
        if (actions.shape != (len(states),) or (actions < 0).any()
                or (actions >= self.act_dim).any()):
            raise ValueError("Expected one valid action index for each observation")
        encoded_actions = F.one_hot(actions, self.act_dim).to(dtype=states.dtype)
        return torch.cat((states, encoded_actions), dim=-1), unbatched

    def errors(self, states, actions, target=False):
        """Return current and frozen-initial mean squared feature errors."""
        inputs, unbatched = self.action_inputs(states, actions)
        with torch.no_grad():
            reference = self.random_target(inputs)
            initial_error = (self.initial_predictor(inputs) - reference).square().mean(dim=-1)
        network = self.target_predictor if target else self.predictor
        with torch.set_grad_enabled(torch.is_grad_enabled() and not target):
            error = (network(inputs) - reference).square().mean(dim=-1)
        if unbatched:
            return error.squeeze(0), initial_error.squeeze(0)
        return error, initial_error

    def values_from_errors(self, error, initial_error):
        return (self.q_scale / math.log(10.0)) * (
            torch.log(initial_error + self.error_epsilon)
            - torch.log(error + self.error_epsilon)
        )

    def selected_values(self, states, actions, target=False):
        return self.values_from_errors(*self.errors(states, actions, target=target))

    def q_values(self, states, target=False):
        parameter = next(self.predictor.parameters())
        states = torch.as_tensor(states, dtype=parameter.dtype, device=parameter.device)
        unbatched = states.ndim == 1
        if unbatched:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[1] != self.obs_dim:
            raise ValueError("Expected one observation or a batch of observation vectors")
        expanded_states = states[:, None, :].expand(-1, self.act_dim, -1)
        actions = torch.arange(self.act_dim, device=states.device).repeat(len(states))
        values = self.selected_values(
            expanded_states.reshape(-1, self.obs_dim), actions, target=target
        ).reshape(len(states), self.act_dim)
        return values.squeeze(0) if unbatched else values

    def forward(self, states):
        return self.q_values(states)


def encode_observation(obs, elapsed_steps, max_episode_steps, observation_input="raw_with_time"):
    """Keep raw continuous coordinates, optionally appending elapsed time."""
    state = torch.as_tensor(obs, dtype=torch.float32).detach().clone()
    if state.ndim != 1 or not torch.isfinite(state).all():
        raise ValueError("Expected a finite observation vector")
    if observation_input == "raw":
        return state
    if observation_input != "raw_with_time" or max_episode_steps <= 0:
        raise ValueError("Expected raw or raw_with_time and a positive episode limit")
    return torch.cat((state, state.new_tensor([elapsed_steps / max_episode_steps])))


@torch.no_grad()
def choose_action(critic, state, rng, epsilon=0.0):
    """Epsilon-greedy actions; equal Q-values select the lowest action index."""
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between zero and one")
    if epsilon and rng.random() < epsilon:
        return int(rng.integers(critic.act_dim))
    values = critic(state)
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise FloatingPointError("Action selection requires finite Q-values")
    return int(values.argmax().item())


@torch.no_grad()
def update_target(critic, tau):
    """Polyak update only the lagged trainable predictor."""
    if not math.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("Target rate must satisfy 0 < tau <= 1")
    for source, destination in zip(critic.predictor.parameters(), critic.target_predictor.parameters()):
        if source.data_ptr() == destination.data_ptr():
            raise ValueError("The target predictor must have separate parameters")
        destination.mul_(1.0 - tau).add_(source, alpha=tau)
