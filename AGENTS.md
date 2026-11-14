# Working instructions

## Repository permissions

Prepare changes locally. Commit, push, publish releases, or change repository
settings only with explicit permission for that action. Never change visibility
or sensitive settings. Do not bring private repository names, paths, history,
unrelated experiments, or credentials into this repository.

## Research and writing

- The research formulation and empirical work belong to Jaeyoung Oh. Do not
  add assistant or AI authorship, acknowledgments, or ownership claims without
  an explicit request. Keep literature references separate from research credit.
- Use professional, precise research prose. The README has a plain author line
  once at the top; do not repeat personal attribution in the methods or results.
- Never use em dashes. Do not reproduce emotional or profane language.
- Keep the name Error-Defined Neural Q-Learning. Describe the signed logarithmic
  initial/current prediction-error ratio as Q itself, trained with Bellman
  targets. It is not an exploration bonus or a gate on a separate Q estimator.
- Distinguish exact zero initialization from preservation of zero on untrained
  inputs after learning. The latter is not guaranteed.
- Report 422/500 held-out landings (84.4%) and mean raw return 132.20 as results
  of one monitor-selected checkpoint. Five evaluation streams are not five
  training seeds. Keep the final checkpoint's 36/100 result separate.
- Do not claim worldwide priority, sustained convergence, conventional
  200-point mean-return success, or an isolated architectural advantage from
  these measurements.

## Scientific behavior

- Preserve continuous raw observation inputs plus elapsed episode time, action
  one-hot encoding, the four-network structure, Q mapping, and update order.
- Learning rewards are zero on nonterminal steps, +100 on native landing, and
  -100 on native crash. Raw Gymnasium returns include shaping and are reporting
  only. Do not use the undefined phrase "task rewards".
- Retain the 1,000-step horizon and exclude entire timeout trajectories from
  training. Native termination at the horizon takes precedence. Timeouts remain
  in evaluation denominators. There are no time or timeout penalties.
- Preserve reverse chronological minibatches followed by one uniform replay
  update per episode, including replay from older accepted episodes after a
  timeout. Do not add outcome balancing, intrinsic rewards, or discretization.
- Continue checkpoints by restoring networks, Adam, and exploration controller.
  Replay and environment/random streams restart. This is not an exact resume.
- Keep recorded results and the bundled checkpoint immutable. Put new runs in
  separate directories. Do not launch full training to check documentation.

## Maintenance

- Keep this a focused research implementation. Do not add CI, permanent test
  scaffolding, analytics services, generic orchestration frameworks, or unused
  compatibility code. Perform bounded verification outside the repository.
- Use descriptive function names without a leading single underscore. Standard
  Python dunder methods are fine.
- Update README and methods documentation with changes to behavior, settings,
  commands, or results. Keep dependent installation and reproduction commands
  accurate. Preserve scientific behavior when cleaning source code.
- The bundled checkpoint and evaluation records predate standalone packaging.
  Packaging checks are not a new training replication.
