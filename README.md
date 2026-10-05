# Brain-aligned recurrent networks for working memory

Status as of 3 October 2026, the date of the last training run and result update. Work in progress: the last training arms are unfinished, and
several analyses have not yet been run on all trained networks.

## What the project asks

Many different recurrent networks can solve a working-memory task. This project asks which
biologically motivated design choices, if any, make a trained network represent memory the
way human neurons do. Networks are trained only on task labels or reward, on the same visual
Sternberg task that patients performed during single-neuron recordings (remember one to three
pictures, wait through a blank delay, report whether a probe picture was among them). Human
neural activity is never a training target. After training, each frozen network is shown the
exact picture sequences from the recording sessions and its internal states are compared with
the recorded firing rates by representational similarity analysis (RSA): the pattern of
pairwise distances between task conditions in the network is correlated with the same pattern
in the neurons, so no unit-to-unit mapping is needed. Behaviour is necessary but is analysed
separately from neural alignment.

## How it is built

**Task.** A trial has fixation (3 model steps), encoding (15 steps per picture, so 15 x load),
a blank delay (15 steps), a probe (10 steps), feedback (1 step) and an inter-trial interval
(2 steps); load is 1, 2 or 3. About half the probes were in the remembered set; a fraction
(target 0.20) of out-of-set probes are lures from the same category as a remembered picture.
A curriculum moves from load 1, no lures and a short delay to the full distribution.
Evaluation uses held-out pictures and reports Wilson 95% intervals.

**Input.** On every step the network receives the 512-dimensional feature vector of a frozen
ImageNet-pretrained ResNet-18 (never fine-tuned) plus a 10-entry task cue (task family,
encoding position, epoch). During the delay the picture input is blank, so all memory lives in
the recurrent core.

**Models.** The core is a gated recurrent unit (GRU); the front end and the action/value
heads are shared, so only the core varies. Five binary mechanisms are switched on or off in a
15-condition design (baseline, all five on, five add-one, five remove-one, three
pairwise-interaction cells):

- *Structure (S):* a locality-masked worker network that runs every step, plus a dense manager
  that updates on a slower clock and sends top-down context to the worker.
- *Modulation (M):* a gate bias driven by accumulated surprise (reward-prediction error at
  feedback, self-generated uncertainty otherwise).
- *Plasticity (P):* differentiable Hebbian fast weights added to the slow trained weights,
  reset every trial.
- *Topography (T):* a smoothness loss that makes neighbouring units on a 2-D sheet similar.
- *Dale's law (D):* a soft penalty pushing each unit's outgoing weights toward one sign
  (80% excitatory, 20% inhibitory).

Each cell is trained with eight seeds under two training signals: supervised (SUP,
cross-entropy) and reinforcement learning (RL, policy gradient). Additional arms: three
rate-based recurrent substrates (excitatory/inhibitory, dynamic-synapse, low-rank); a
locality-matched flat control and a crossed width x connectivity x topography set that
separate hierarchy from recurrent sparsity; a memory-demand arm with occasional delayed
category reports; four local-learning cells (node perturbation with an e-prop fallback,
no backpropagation through time); and a continuation study that trains matched checkpoints for
24,000 further updates either on working memory alone or cycling six NeuroGym tasks. Capacity
is matched at the synapse level: 73,728 effective structural synapses for the flat 128-unit
core against 74,052 for the 196-unit worker plus 24-unit manager (0.44% apart).

**Neural data (public, DANDI).** Datasets 000469 and 000673: human medial temporal lobe and
medial frontal cortex single units during picture Sternberg (data descriptor
https://pmc.ncbi.nlm.nih.gov/articles/PMC10796636/; associated paper
https://doi.org/10.1038/s41586-024-07309-z). Together: 57 patients, 65 sessions, 2,298
neurons, 8,820 replayed trials. Regions: hippocampus and amygdala (medial temporal lobe, MTL);
dorsal anterior cingulate, pre-supplementary motor area and ventromedial prefrontal cortex
(medial frontal cortex, MFC). Patients are split into 38 discovery and 19 confirmation
patients (`configs/patient_folds.json`); both splits have been inspected, so the confirmation
split is no longer untouched. Dataset 001187 (46 sessions, 39 patients, 6,439 trials, 941
hippocampal and amygdala units) is the planned independent confirmation set; 37 of its
patient-session pairs (31 patients) also appear in 000673 and are handled through a
patient-session registry.

**Measurements.** Maintenance RSA: within each session and load, cross-validated Mahalanobis
(crossnobis) distances between remembered-set conditions, model versus neurons, Spearman
correlation, averaged over sessions; normalised by the split-half reliability of the neural
geometry (mean 0.643). Probe RSA: 12 conditions (load x probe-in-set x answered correctly) on
a pseudopopulation across sessions. Intervals resample patients, not sessions or trials.
Controls: frozen-encoder-only geometry, label-only task-structure geometry, an untrained
network, and a planted-signal sensitivity curve. Supporting analyses: cross-temporal decoding,
condition-marginalised PCA (not a full demixed-PCA fit), fixed points, graph measures, and
interventions that disrupt hidden activity and fast-weight traces separately.

## Progress (as of 3 October 2026)

**Trained.** 392 networks are available for replay: 192 core runs (120 SUP = 15 cells x 8
seeds; 72 RL = 7 cells x 8 seeds plus 8 hierarchical cells x 2 seeds), 32 local-learning runs
and 168 control runs (48 rate substrates, 24 memory-demand, 16 locality-matched flat, 80
crossed flat). The continuation study is 38 of 64 runs complete. Alignment has been computed
for the 192 core runs; the other 200 await replay.

**Behaviour.** SUP networks (mean of 15 cell means, 120 runs) score 0.985 / 0.955 / 0.926 at
loads 1 / 2 / 3. Patients pooled over sessions score 0.946 (7,184 trials), 0.887 (891 trials)
and 0.907 (7,184 trials). RL cells without hierarchy reach load-3 accuracy 0.71 to 0.93; the
2-seed hierarchical RL cells reach 0.51 to 0.71. The four local-learning cells (8 seeds each)
stay at chance, 0.49 to 0.51 at every load.

**Alignment with human neurons (192 core runs).**
- Maintenance alignment is close to zero. Mean signed correlation over supervised run-session
  observations is -0.019; cell means range from -0.063 to +0.007. The frozen image-feature
  geometry alone gives +0.025 (pooled region, 65 sessions).
- Probe alignment is larger: raw pooled mean 0.470 (SUP 0.503, RL 0.414), neural ceiling
  0.506. By region it is highest in preSMA (0.415) and MFC (0.397) and lowest in hippocampus
  (0.028). An untrained network scores 0.091 on the normalised version of this measure,
  against 0.923 for the trained runs. The 12 probe conditions are defined by variables that
  also form obvious task structure, and the saved score has not yet been shown to exceed
  load, membership and correctness alone.
- Maintenance and probe differ in estimator (per-session stratified versus pooled
  pseudopopulation) as well as epoch, so their gap is not yet attributed to the epoch; a
  comparison that crosses estimator and epoch is under way.
- Mechanism contrasts at equal training budget, SUP, 8 paired seeds, patient-clustered
  bootstrap (effect in correlation units, 95% interval, bootstrap p). Maintenance RSA, 57
  patients: adding plasticity to the baseline -0.030 (-0.054 to -0.009, p=0.005); full model
  minus the model without hierarchy +0.040 (0.012 to 0.069, p=0.001). Probe RSA: adding
  plasticity -0.038 (-0.058 to -0.018, p<0.001); full minus no hierarchy +0.049 (0.028 to
  0.071, p<0.001); full minus no modulation -0.009 (-0.016 to -0.003, p=0.004). These are
  differences of a few hundredths against a ceiling of 0.643, and the model without
  hierarchy has higher load-3 accuracy (0.948 versus 0.930), so alignment and accuracy do not
  move together. RL contrasts rest on two seeds of poorly performing hierarchical cells and
  are exploratory.
- Does maintenance alignment predict robustness? Among 150 accuracy-matched runs (within 0.02
  of median accuracy 0.951), signed maintenance alignment versus longer-delay accuracy:
  Pearson r = 0.127 (-0.051 to 0.280, p=0.120); versus accuracy at novel load 4: r = 0.053
  (-0.146 to 0.229, p=0.523). Minimum detectable |r| at n=150 is 0.227. The sample is
  bounded by alignment coverage and grows only after replay of the other runs.
- Planted-signal check (pooled region, 56 patients, 65 sessions): a synthetic model side with
  no shared condition geometry returns 0.127 (0.075 to 0.182); a full copy of the neural
  geometry returns 0.446 (0.396 to 0.492). The estimator therefore has a positive offset on
  synthetic data, and real network values (mean -0.019) lie below it; what it can detect near
  zero is still being worked out, and no claim about maintenance geometry rests on it yet.

**Where memory is held.** Hidden-activity and fast-weight-trace disruptions were applied
separately (1,080 accuracy rows: 72 checkpoints x 5 conditions x 3 loads). During the delay,
disrupting the synaptic trace moves the targeted carrier about 26 times more than the other
(off-to-on deviation ratio 0.039); disrupting activity before encoding gives 0.107. Delay
disruptions cost 0.15 to 0.46 accuracy against 0.00 to 0.05 before encoding. The
delay-activity condition (ratio 0.72) is confounded by drift, because holding the trace
frozen lets the unlesioned trace keep being written; it is recorded as a limitation.

**Continuation study (SUP, 38 of 64 runs, 4 to 5 seeds per cell so far, interim).** After
24,000 further updates, Sternberg load-3 accuracy is 0.921 to 0.946 when working memory alone
is trained and 0.545 to 0.612 when six tasks are cycled; load-1 accuracy is 0.99 versus 0.85
to 0.96.

**Measurement issues found and handled.** One nonpositive maintenance duration in 000469
(session SBID_9_P34CS, trial 28, load 1) was quarantined, and replay now keeps source-trial
indices. The cross-temporal stability and persistence tables come from an earlier estimator
and are being regenerated; their old values are not interpreted. At equal nominal weight the
Dale's-law penalty is about 3.1 times stronger on hierarchical than flat cells (0.0692
versus 0.0221), and the flat core's initial recurrent gain is about three times the worker's
(0.608 versus 0.201); both are stated as alternative accounts of any architecture effect.

## What is running and what is next

Nothing is training now: the continuation run stopped at 38 of 64 and resumes from
checkpoints. In order:
1. Finish the remaining 26 continuation runs, then the 32-run from-scratch six-task arm
   (estimated at about four days).
2. Replay all trained networks on the recorded trials (about 70 to 95 GB of activity logs).
3. Alignment for the control arms, to test whether hierarchy's effect survives matched
   recurrent sparsity and width; per-population alignment (worker and manager separately,
   never pooled) under both training signals; distributions by load, region and patient
   behind every mean; then rerun the robustness regression on the larger set.
4. Full-delay maintenance RSA over the whole recorded delay (mean 2.69 s) instead of a fixed
   1.5 s window; the first rows are written, the rest are pending.
5. Probe RSA against the task-structure and encoder baselines with held-out semipartial
   correlation; equal-performance mechanism contrasts.
6. Confirmation on dataset 001187 after the analysis plan is frozen.

## Installation

Python 3.11 or newer. Install a PyTorch build for your hardware first (`make setup` installs
CUDA 12.8 wheels; developed on an RTX 5070 Ti with 12 GB), then:

```bash
python -m pip install torch torchvision
python -m pip install -e ".[analysis,dev]"
python -m pip install -e ".[multitask]"   # only for the six-task NeuroGym arm
```

Paths for data, stimuli, caches, checkpoints and results are in `configs/paths.yaml`; override
with `BRAINALIGN_WM_PATHS_CONFIG`, or `BRAINALIGN_WM_DATA_ROOT`,
`BRAINALIGN_WM_STIMULI_ROOT`, `BRAINALIGN_WM_RESULTS_ROOT`,
`BRAINALIGN_WM_FEATURE_CACHE_ROOT`, `BRAINALIGN_WM_ACTIVITY_LOGS_ROOT`. Inspect the resolved
configuration with `python -m brainalign_wm.config`. Neural recordings are public on DANDI;
recordings, stimulus images, caches, checkpoints and results are not distributed here.

## Tests and main commands

```bash
make test             # unit tests; 486 passed at the last full run, 13 September 2026
make audit-campaign   # consistency check of run records, resolved configs and metrics
make fetch-encoder && make features   # ResNet-18 weights and cached picture features
make recovery         # simulated-geometry recovery check; must pass before real-data alignment
make run-grid         # supervised ablation grid; `make run-grid-demo` runs without PyTorch
python scripts/run_continuations.py --execute                           # continuation training
python scripts/run_multitask_from_init.py --supervision SUP --execute   # six-task from scratch
./replay.sh           # write activity logs and run post-training analysis
python scripts/run_measurement_outputs.py --dry-run   # list full-delay, baseline and distribution work units
```

Training and replay are resumable. `train.sh`, `multitask.sh` and `replay.sh` run one
detached process at a time; `remaining.sh` shows what is left and `stop.sh` stops the running one.
Run `make help` for all shortcuts.

## Repository layout

| Path | Contents |
| --- | --- |
| `brainalign_wm/models/` | Flat, hierarchical, rate-based and vanilla recurrent cores; shared front end and heads |
| `brainalign_wm/mechanisms/` | Reflective gate and local-learning rules |
| `brainalign_wm/tasks/` | Sternberg generator, curriculum, picture bank |
| `brainalign_wm/training/` | Training, replay of recorded trials, activity logging |
| `brainalign_wm/neural/` | Neural-data adapters and simulated-neural-data tools |
| `brainalign_wm/analysis/` | RDMs, RSA, contrasts, persistence, geometry, fixed points, measurement validation, full-delay and distribution analyses |
| `scripts/` | Experiment, audit and analysis commands |
| `configs/` | Experiment, path and patient-fold configuration |
| `tests/` | Automated tests |
| `run_grid.py` | Ablation-grid orchestrator |
