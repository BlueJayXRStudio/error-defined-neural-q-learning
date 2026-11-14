# Method and experimental protocol

## Error-defined action values

The state representation contains LunarLander-v3's eight continuous observation
components and elapsed episode steps divided by 1,000. Appending a four-way
one-hot action gives a 13-dimensional network input. Observations are neither
discretized nor normalized.

Let $f(x)$ be a frozen random feature network, $g_0(x)$ a frozen initial
predictor, and $g_\theta(x)$ the trainable predictor, each with $d=128$ outputs.
Define

$$
\begin{aligned}
e_0(x) &= \frac{1}{d}\left\lVert g_0(x)-f(x)\right\rVert_2^2, \\
e_\theta(x) &= \frac{1}{d}\left\lVert g_\theta(x)-f(x)\right\rVert_2^2.
\end{aligned}
$$

$$
Q_\theta(x)=c\log_{10}\!\left(\frac{e_0(x)+\eta}{e_\theta(x)+\eta}\right),
\qquad c=100,\quad\eta=10^{-8}.
$$

The live predictor initially equals $g_0$, giving exactly zero Q at every
input. Error reduction gives positive Q; error increase gives negative Q.
The fixed reference is evaluated at each input throughout training. It does
not prevent generalization from changing values at other inputs.

The implementation computes a difference of logarithms. With the positive
numerical floor $\eta$, Q has an input-dependent upper bound
$c\log_{10}((e_0+\eta)/\eta)$. Negative values are retained. The frequently useful
one-tenth-error and ten-times-error examples give approximately +100 and -100,
respectively, when the floor is negligible.

All feature networks have two 64-unit ReLU hidden layers and 128 outputs.
Only $g_\theta$ is optimized. Its 13,376 parameters form one of four networks:
the live predictor, frozen random target, frozen initial predictor, and a lagged
target predictor. Together they contain 53,504 parameters.

## Bellman regression

The lagged predictor defines $Q_{\bar\theta}$ using the same permanently frozen
references. For a transition $(s,a,r,s')$, the detached target is

$$
y = \begin{cases}
r, & \text{native termination},\\
r+\gamma\max_{a'}Q_{\bar\theta}(s',a'), & \text{otherwise}.
\end{cases}
$$

Adam minimizes

$$
\mathcal L(\theta)=\frac{1}{2B}\sum_{i=1}^{B}
\left(\frac{Q_\theta(s_i,a_i)-y_i}{100}\right)^2.
$$

The division by 100 scales the optimization loss; it does not change rewards
or the Bellman equation. Targets have no gradient path. There is no separate
objective that unconditionally minimizes feature prediction error. An increase
in Q corresponds to a decrease in error, and a decrease in Q corresponds to an
increase in error. Shared parameters can couple these effects across samples.

After every optimizer step, the lagged predictor receives the update
$\bar\theta \leftarrow 0.995\bar\theta+0.005\theta$. Both reference networks
remain unchanged. This also applies between successive reverse-replay batches.

| Setting | Recorded value |
| --- | --- |
| Discount factor | 0.99 |
| Adam learning rate | 0.0003 |
| Adam betas, epsilon | (0.9, 0.999), 1e-8 |
| Predictor gradient norm limit | 10 |
| Minibatch size | 32 |
| Replay capacity | 100,000 transitions |
| Uniform replay updates | One after each episode when replay is nonempty |
| Local epoch threshold | 64,000 interactions, finishing the current episode |
| Training environments | One |
| Device and threads | CPU, one PyTorch thread |

## Rewards, episode handling, and replay

Learning rewards are zero on every nonterminal transition. Native terminal
rewards are +100 for landing and -100 for crash. Raw environment rewards include
shaping and engine costs, and are accumulated separately for reporting. No
intrinsic reward, time cost, or timeout penalty enters the learning signal.

Episodes end at native termination or the 1,000-step horizon. Native termination
at the horizon takes precedence. An entire timed-out trajectory is excluded from
learning, including all its nonterminal transitions. Excluding timeouts is a
substantive part of this experimental protocol.

Network weights are fixed during each episode. After a native terminal outcome,
all transitions enter a FIFO replay buffer and are replayed in reverse
chronological minibatches. Each minibatch uses one-step Q-learning targets, not
full Monte Carlo returns. A batch is optimized jointly, so this is not a serial
transition-by-transition tabular sweep. One uniform minibatch sampled without
replacement follows each episode, including timeout episodes if earlier
accepted data exist. Successful and failed episodes are not balanced or
permanently retained.

## Training and evaluation protocol

The recorded lineage used three consecutive training segments:

| Segment | Initialization | RNG seed | Wall budget | Added training interactions | Cumulative completed epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Fresh | 101 | 40 minutes | 14,312,348 | 222 |
| 2 | Segment 1 final checkpoint | 104 | 40 minutes | 14,329,732 | 444 |
| 3 | Segment 2 final checkpoint | 105 | 80 minutes | 29,098,107 | 895 |

All four networks, Adam, and the complete exploration-controller state carry
over between segments. Replay and environment/action/replay random streams
restart. A partially completed epoch restarts with a full local step budget.
The cumulative interaction and update counts include that partial work.

The controller alternates between epsilon-greedy exploration at epsilon 0.10
and greedy behavior at epsilon zero. In the exploration phase, a 10% random-action
override samples uniformly among all four actions; otherwise the policy chooses
greedily. Greedy ties select the lowest action index. An episode
uses one epsilon value and one fixed policy throughout collection.

Frozen greedy monitoring occurs at initialization and every ten completed
epochs, using a separate environment and seeds 2,000,001 through 2,000,050.
Monitoring seeds each episode individually. It contributes no replay data and
does not update the model. Its time counts against the segment budget, while
its interactions are recorded separately. Continuations preserve controller
history without repeating the initialization monitor.

| Controller parameter | Value |
| --- | --- |
| Minimum phase length | 10 epochs |
| Maximum exploration phase | 30 epochs |
| Meaningful improvement | 5 percentage points |
| Substantial decline | 15 percentage points |
| Confirmation | Two consecutive monitoring checks |
| Plateau patience | Three checks |
| Landing rate exempt from plateau switching | 80% |

Exploration ends after a confirmed improvement over its entry score or after
the maximum exploration duration. Greedy behavior switches to exploration
after a confirmed decline or a plateau below 80%. Meaningful improvement updates
the phase reference. These thresholds are experimental choices, not established
optimal settings. The exact configurations are in [configs/](../configs/).

Training budgets are checked at episode/replay boundaries. Frozen monitoring
also checks the deadline between episodes. Hardware speed, checkpoint I/O, and
runtime differences can alter the number of interactions within a time budget.
Repeating the procedure does not promise bitwise reproduction of training.

## Checkpoint selection and holdout

Epoch 810 was the unique best completed monitoring checkpoint in the last
segment and across the recorded lineage, scoring 38/50 landings. It contains
52,223,579 training interactions and 539,297 optimizer updates. It was fixed
before the holdout seed streams were evaluated.

The holdout creates one environment for each seed 3,000,001 through 3,000,005.
Only the first reset receives that seed. The remaining 99 resets use
`seed=None`, advancing the same environment RNG stream. Every episode uses
greedy actions, frozen weights, and the 1,000-step horizon. Timeouts remain in
the denominator. This differs from the monitor and earlier evaluation banks,
which reset with a separate seed per episode.

The holdout records 422 landings, 15 crashes, and 63 timeouts across 500 episodes.
Mean raw environment return is 132.2033; mean sparse terminal return is 81.4.
The five streams evaluate one policy, not five independently trained agents.

The earlier evaluation bank, seeds 1,000,001 through 1,000,100, gives 88/100
landings for epoch 810 and 36/100 for the last training checkpoint. Five further
per-episode-seed banks give 413/500 landings. These records are separate from
the stream-based holdout and are grouped in
[evaluation_banks.json](../results/evaluation_banks.json).

The selected checkpoint is included unchanged. SHA-256:

```text
bdcc186d447c377038ffa164dc16097e2cecbc2559e5208f05cded98c74422b7
```

## Interpretation and reproducibility

The observed result is repeated greedy landing under terminal-only learning
rewards from a single neural training lineage. It does not isolate the error
representation from replay, timeout exclusion, exploration scheduling, or
checkpoint selection. Matched conventional Q-network comparisons, independent
training seeds, and other environments are needed to investigate those questions.

The data were collected before standalone packaging. Core network and TD-update
behavior are preserved in the packaged implementation. The release removes
unused exploration schedules and development-only recording and reorganizes
imports and command-line entry points. The checkpoint remains byte-identical;
numeric evaluation records are retained. The artifact manifest describes export
format changes, and the lineage file retains settings and implementation hashes
from the recorded experiments.

The neural formulation was developed following experiments in sparse-reward
tabular Q-learning. Reverse replay was developed independently during that work;
Lin (1992) provides a direct prior publication of backward replay. Research
precedents are cited in the [README](../README.md#related-work).
