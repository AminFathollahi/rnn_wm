"""Item 10.2 (comments.txt §5, fixes A5): a sanity gate the local-learning
(L=1) rule must clear BEFORE any M**L cell is trained. Every M**L run
previously sat at accuracy 0.5/0.35/0.475 -- 0.35 is BELOW chance (0.50) --
and the protocol blamed node-perturbation variance scaling. That excuse
does not survive: the task here is a trivial 1-tick one-hot -> correct-
action mapping (no recurrence, no delay, no partial observability) that a
single `NodePerturbationLearner` on one `nn.Linear(8,3)` must be able to
solve. If it can't solve THIS, the M**L results are an implementation
artefact, not a scientific finding.

FIRST RUN (this test unmodified, at the then-production
`configs/config.yaml` defaults `perturb_sigma=0.05, lr_local=1e-3`):
FAILED, mean acc 0.365 over 3 seeds -- matching the broken M**L numbers.
Root-caused (not a learning-rate scaling issue -- sweeping `lr_local` up
1000x at `sigma_p=0.05` never left ~0.26, still below chance): at
`sigma_p=0.05`, sampling `xi ~ N(0, sigma_p^2)` on top of a freshly
`nn.Linear`-initialized layer flips the discrete argmax decision only
~0.2% of the time (measured directly), because the perturbation is
swamped by the ~0.39-average natural top1-top2 logit gap at init. Node
perturbation's entire signal is the correlation between xi and reward;
if xi almost never changes which action is taken, that correlation is
essentially unmeasurable regardless of `lr_local` -- there is no
exploration to learn from, and no rescaling of the update creates it.
Fix applied: raised `configs/config.yaml`'s `perturb_sigma` (0.05->0.3)
and jointly re-tuned `lr_local` (1e-3->0.5) via a grid sweep, confirmed
by a directly-measured flip rate (~14% at sigma_p=0.2, higher at 0.3) and
by this test now passing at those values. This test uses those same
corrected production defaults (not test-only numbers) so it exercises
what `train.py` actually configures.

Do NOT tune this test further to pass (comments.txt is explicit: either
outcome -- pass or fail -- is a valid result to report); the values below
are the corrected production config, not a test-specific fudge."""
import torch
import torch.nn as nn

from brainalign_wm.mechanisms.local_learning import NodePerturbationLearner

N_INPUTS = 8
N_ACTIONS = 3
TARGET = torch.arange(N_INPUTS) % N_ACTIONS  # fixed one-hot-index -> correct-action assignment


def _run(seed: int, n_trials: int = 5000, batch_size: int = 20,
         sigma_p: float = 0.3, lr_local: float = 0.5) -> float:
    """Trains a plain `nn.Linear(8, 3)` via a rung-1 `NodePerturbationLearner`
    on the 1-tick one-hot -> correct-action task, returns accuracy over the
    FINAL 500 trials (a converged-performance readout, not a lifetime
    average). No autograd, no BPTT -- perturb the linear output, trace
    against the input (this has no recurrence, so `h_prev` in `trace_step`'s
    signature is just the current input, same convention `_run_trial_local`
    already uses for `heads.pi`/`heads.value` -- plain Linear heads with no
    "previous hidden state" of their own)."""
    torch.manual_seed(seed)
    linear = nn.Linear(N_INPUTS, N_ACTIONS)
    learner = NodePerturbationLearner(
        weight_hh=linear.weight, weight_ih=None,
        sigma_p=sigma_p, gamma_e=0.9, lr_local=lr_local, seed=seed,
    )
    rng = torch.Generator().manual_seed(seed + 1000)

    correct_history: list[bool] = []
    n_batches = n_trials // batch_size
    with torch.no_grad():
        for _ in range(n_batches):
            idx = torch.randint(0, N_INPUTS, (batch_size,), generator=rng)
            x = torch.nn.functional.one_hot(idx, N_INPUTS).float()
            target = TARGET[idx]

            logits = linear(x)
            xi = learner.sample_perturbation(logits.shape)
            perturbed = logits + xi
            action = perturbed.argmax(dim=-1)
            reward = (action == target).float()

            learner.trace_step(xi, h_prev=x, x_t=x)
            learner.apply_update(reward)
            correct_history.extend(reward.bool().tolist())

    tail = correct_history[-500:]
    return sum(tail) / len(tail)


def test_node_perturbation_learns_1tick_one_hot_task_above_0_95():
    accs = [_run(seed=s) for s in range(3)]
    mean_acc = sum(accs) / len(accs)
    assert mean_acc > 0.95, (
        f"NodePerturbationLearner (rung 1) failed the item 10.2 gate: "
        f"mean accuracy {mean_acc:.3f} over seeds {accs} on a trivial "
        f"1-tick one-hot task within 5000 trials -- local-learning results "
        f"elsewhere in this repo are an implementation artefact until this "
        f"passes (comments.txt item 10.2). Do not tune this test to pass; "
        f"fix the learner."
    )
