"""Alternate exploration and greedy behavior using frozen greedy monitoring scores."""

import math


def validate_switching_config(config):
    """Validate the optional controller settings without changing the configuration."""
    settings = config.get("epsilon_switching")
    fields = {
        "monitor_every_epochs", "monitor_episodes", "monitor_seed", "min_phase_epochs",
        "max_exploration_epochs", "improvement_delta", "drop_delta", "patience_checks",
        "confirmation_checks", "healthy_rate",
    }
    if not isinstance(settings, dict) or set(settings) != fields:
        raise ValueError(f"epsilon_switching requires exactly {sorted(fields)}")
    for name in ("monitor_every_epochs", "monitor_episodes", "min_phase_epochs",
                 "max_exploration_epochs", "patience_checks", "confirmation_checks"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"epsilon_switching.{name} must be a positive integer")
    if type(settings["monitor_seed"]) is not int or settings["monitor_seed"] < 0:
        raise ValueError("epsilon_switching.monitor_seed must be a nonnegative integer")
    cadence = settings["monitor_every_epochs"]
    if settings["min_phase_epochs"] % cadence or settings["max_exploration_epochs"] % cadence:
        raise ValueError("Minimum phase and maximum exploration epochs must be multiples of monitoring cadence")
    if settings["max_exploration_epochs"] < settings["min_phase_epochs"]:
        raise ValueError("Maximum exploration duration cannot be shorter than the minimum phase")
    for name in ("improvement_delta", "drop_delta", "healthy_rate"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"epsilon_switching.{name} must be finite and in (0,1]")
    first = settings["monitor_seed"]
    last = first + settings["monitor_episodes"] - 1
    if first <= 1000100 and last >= 1000001:
        raise ValueError("Monitoring seeds must not overlap final evaluation seeds 1000001 through 1000100")
    if config.get("epsilon_decay_epochs", 0) or config.get("epsilon_decay_seconds", 0) or config.get("epsilon_landing_decay") is not None:
        raise ValueError("Epsilon switching cannot combine with time, epoch, or landing-triggered decay")
    epsilon = config.get("epsilon")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Epsilon switching requires an exploration epsilon in (0,1]")
    return settings


class ExplorationController:
    """Use phase-specific thresholds; monitoring never changes the learner itself."""

    def __init__(self, config):
        self.settings = dict(validate_switching_config(config))
        self.exploration_epsilon = config["epsilon"]
        self.mode = "exploration"
        self.phase_start_epoch = 0
        self.baseline_score = None
        self.best_score = None
        self.plateau_checks = 0
        self.dip_checks = 0
        self.improvement_checks = 0
        self.last_observed_epoch = None
        self.last_score = None
        self.observations = 0
        self.switches = 0

    def epsilon(self):
        return self.exploration_epsilon if self.mode == "exploration" else 0.0

    def state_dict(self):
        """Return a detached JSON-serializable account of the controller state."""
        return {
            "mode": self.mode, "phase_start_epoch": self.phase_start_epoch,
            "baseline_score": self.baseline_score, "best_score": self.best_score,
            "plateau_checks": self.plateau_checks, "dip_checks": self.dip_checks,
            "improvement_checks": self.improvement_checks,
            "last_observed_epoch": self.last_observed_epoch, "last_score": self.last_score,
            "observations": self.observations, "switches": self.switches,
        }

    def load_state_dict(self, state, completed_epochs):
        """Restore a validated checkpoint state without resetting phase or patience."""
        if type(completed_epochs) is not int or completed_epochs < 0:
            raise ValueError("Completed epochs must be a nonnegative integer")
        if not isinstance(state, dict) or set(state) != set(self.state_dict()):
            raise ValueError("Controller checkpoint fields differ")
        if state["mode"] not in ("exploration", "greedy"):
            raise ValueError("Invalid controller mode")
        for name in ("phase_start_epoch", "plateau_checks", "dip_checks", "improvement_checks",
                     "observations", "switches"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"Invalid controller counter: {name}")
        cadence = self.settings["monitor_every_epochs"]
        start, last = state["phase_start_epoch"], state["last_observed_epoch"]
        if last is None:
            fresh = ExplorationController({"epsilon": self.exploration_epsilon,
                                           "epsilon_switching": self.settings}).state_dict()
            if completed_epochs != 0 or state != fresh:
                raise ValueError("Unobserved controller state must be a fresh initial state")
        else:
            if type(last) is not int or not 0 <= start <= last <= completed_epochs:
                raise ValueError("Controller epoch counters are inconsistent")
            if start % cadence or last % cadence:
                raise ValueError("Controller epochs do not follow the monitoring cadence")
            for name in ("baseline_score", "best_score", "last_score"):
                value = state[name]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid controller score: {name}")
            observations, switches = state["observations"], state["switches"]
            if not 1 <= observations <= last // cadence + 1 or switches >= observations:
                raise ValueError("Controller observation and switch counters are inconsistent")
            expected_mode = "greedy" if switches % 2 else "exploration"
            if state["mode"] != expected_mode or bool(start) != bool(switches):
                raise ValueError("Controller phase does not match its switch count")
            if start < switches * self.settings["min_phase_epochs"]:
                raise ValueError("Controller switches violate the minimum phase duration")
            phase_checks = min((last - start) // cadence, observations - 1)
            if any(state[name] > phase_checks for name in ("plateau_checks", "dip_checks", "improvement_checks")):
                raise ValueError("Controller patience exceeds observations in the current phase")
            if state["best_score"] < state["baseline_score"]:
                raise ValueError("Best controller score cannot be below its phase baseline")
            if last == start and not state["baseline_score"] == state["best_score"] == state["last_score"]:
                raise ValueError("A new controller phase must start from its observed score")
            if state["dip_checks"] and state["last_score"] > state["best_score"] - self.settings["drop_delta"] + 1e-12:
                raise ValueError("Controller drop counter does not match the latest score")
            if state["improvement_checks"] and state["last_score"] + 1e-12 < state["baseline_score"] + self.settings["improvement_delta"]:
                raise ValueError("Controller improvement counter does not match the latest score")
            if state["mode"] == "greedy" and state["last_score"] >= self.settings["healthy_rate"] and state["plateau_checks"]:
                raise ValueError("A healthy greedy observation must reset plateau patience")
        # All checks precede mutation, so rejected states leave the controller intact.
        for name, value in state.items():
            setattr(self, name, value)

    def observe(self, completed_epochs, success_rate):
        """Consume one complete greedy monitor; an epoch-zero baseline is required first."""
        if type(completed_epochs) is not int or completed_epochs < 0:
            raise ValueError("Completed epochs must be a nonnegative integer")
        if completed_epochs % self.settings["monitor_every_epochs"]:
            raise ValueError("Monitoring epoch must follow the configured cadence")
        if type(success_rate) not in (int, float) or not math.isfinite(success_rate) or not 0 <= success_rate <= 1:
            raise ValueError("Monitor landing rate must be finite and in [0,1]")
        if self.last_observed_epoch is None:
            if completed_epochs != 0:
                raise ValueError("An initial monitor at epoch zero is required")
        elif completed_epochs <= self.last_observed_epoch:
            raise ValueError("Monitor epochs must increase strictly")

        before = self.mode
        epsilon_before = self.epsilon()
        phase_age = completed_epochs - self.phase_start_epoch
        reason = "hold"
        if self.last_observed_epoch is None:
            self.baseline_score = self.best_score = success_rate
            reason = "initial_baseline"
        else:
            # Small gains accumulate against the last materially improved score.
            if success_rate + 1e-12 >= self.best_score + self.settings["improvement_delta"]:
                self.best_score = success_rate
                self.plateau_checks = 0
            elif self.mode == "greedy" and success_rate >= self.settings["healthy_rate"]:
                self.plateau_checks = 0
            else:
                self.plateau_checks += 1
            self.dip_checks = (self.dip_checks + 1 if success_rate <= self.best_score - self.settings["drop_delta"] + 1e-12 else 0)
            self.improvement_checks = (self.improvement_checks + 1 if success_rate + 1e-12 >= self.baseline_score + self.settings["improvement_delta"] else 0)
            if phase_age >= self.settings["min_phase_epochs"]:
                if self.mode == "exploration":
                    if self.improvement_checks >= self.settings["confirmation_checks"]:
                        reason = "confirmed_improvement"
                    elif phase_age >= self.settings["max_exploration_epochs"]:
                        reason = "exploration_limit"
                elif self.dip_checks >= self.settings["confirmation_checks"]:
                    reason = "confirmed_drop"
                elif self.plateau_checks >= self.settings["patience_checks"] and success_rate < self.settings["healthy_rate"]:
                    reason = "plateau_below_healthy_rate"

        self.last_observed_epoch = completed_epochs
        self.last_score = success_rate
        self.observations += 1
        decision = {
            "completed_epochs": completed_epochs, "success_rate": success_rate,
            "mode_before": before, "epsilon_before": epsilon_before,
            "reason": reason, "phase_age_epochs": phase_age,
            "baseline_score": self.baseline_score, "best_score": self.best_score,
            "plateau_checks": self.plateau_checks, "dip_checks": self.dip_checks,
            "improvement_checks": self.improvement_checks,
        }
        switched = reason not in ("hold", "initial_baseline")
        if switched:
            self.mode = "greedy" if before == "exploration" else "exploration"
            self.phase_start_epoch = completed_epochs
            self.baseline_score = self.best_score = success_rate
            self.plateau_checks = self.dip_checks = self.improvement_checks = 0
            self.switches += 1
        decision.update(mode_after=self.mode, epsilon_after=self.epsilon(), switched=switched,
                        state=self.state_dict())
        return decision
