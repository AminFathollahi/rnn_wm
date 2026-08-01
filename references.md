# References

Every source used to make a concrete decision in the 2026-07-26 audit and
redesign (`comments.txt`), and the specific idea taken from each. Local
copies live in `../` (the `RNNs/` folder) unless marked *(web only)*.

Grouped by what the reference was used **for**, not alphabetically — the
point of this file is traceability from a design choice back to its source.

---

## 0. The companion paper (digital-twin target, comments.txt §0)

### Fathollahi, M.A. (in prep). *A controllable code for working memory: content and context geometry benchmarked against, and validated by, prefrontal stimulation.*
`../wm_dynamics/PAPER_REPORT.tex` — nine datasets (human single units, iEEG,
ECoG, macaque PFC; seven observational cohorts, two electrical-stimulation
cohorts).

The source of the four findings (C1-C4, pre-registered in
`preregistration.md`'s 2026-07-26 amendment) this RNN is built to reproduce
as a controllable digital twin: content/context rotation asymmetry,
load-invariant low-dimensional manifold, single-trial-only identifiability
of the maintenance dynamics, and a dominant contracting mode whose alignment
predicts real stimulation's causal effect. Every Phase 8 analysis exists to
test one of these four against this repo's RNN.

---

## 1. Parameter count and model size

### Lei, X., Ito, T., & Bashivan, P. (2024). *Geometry of naturalistic object representations in recurrent neural network models of working memory.* NeurIPS 2024. arXiv:2411.02685
`../Geometry of naturalistic object representations in.pdf` — Mila / McGill / IBM.

The single most load-bearing reference in this redesign; it is the closest
published analogue of what this repo is building.

| Taken from it | Used for |
|---|---|
| Architecture: frozen ImageNet **ResNet50** → pointwise conv to RNN hidden size → concat **task-index vector** → FC + **LayerNorm** → RNN → FC → 3 outputs (**match / non-match / no-action**) | Confirms `models/front_end.py` + `models/heads.py` are already the right shape. No change needed — this is the strongest evidence the current front end is publishable as-is. |
| **512 units for vanilla RNN vs 256 units for GRU/LSTM**, chosen explicitly "to ensure comparable model performance as well as comparable model parameters" | The matching rule for the vanilla/GRU/LSTM arm (§Q4 of `comments.txt`). Do **not** compare cell types at equal *unit* count. |
| "performance increased with the number of model parameters, and with identical parameter count, vanilla RNN accuracy was lower compared to their gated counterparts" | Predicts the expected direction of the vanilla-vs-gated result, so a null is interpretable. |
| Behavioral criterion: "**All models reached > 95% accuracy on train and > 90% on validation set with novel object angles**" | The convergence gate (0.90 on held-out). Also the practice of gating on a *generalization* set, not the training distribution. |
| Training diet manipulation: **STSF / STMF / MTMF** (single-task-single-feature, single-task-multi-feature, multi-task-multi-feature) | The whole "WM-only vs general cognitive tasks" arm. Their result — task-irrelevant features are decodable at >85% in multi-task models but not single-task ones — is the pre-registered prediction. |
| Decoders trained on RNN hidden state to predict **task-irrelevant** object properties; cross-task decoder generalization matrices | The "decode non-identity information" analysis the user asked for. Their features are Location/Identity/Category. |
| Finding: gated RNNs (GRU/LSTM) use **task-specific** subspaces, vanilla RNNs use a **shared** subspace across tasks | Why the cell-type arm only pays off *with* the multi-task arm — on a single task there is no cross-task subspace to compare. |
| **Orthogonalization index** *O* = E[triu(1 − |cos(W_i, W_j)|)] over decision-hyperplane normals | The concrete metric for "content/context representations" geometry. |
| "chronological memory subspaces"; "the transformation of WM encodings into memory was shared across stimuli, yet the transformations governing retention ... were distinct across time" | The CTG / rotation analysis framing: encoding→memory transform is stimulus-general, memory→memory transforms are time-specific. |
| Optimizer: AdamW, lr 3e-5, multi-step decay γ=0.1 / 100 iters, batch 256, trials generated on the fly | Sanity check on the repo's Adam @ 3e-4, batch 16 — the repo's batch is 16× smaller. |
| Stimuli: ShapeNet, 4 categories × 2 identities, multiple view angles, **1 of 4 screen locations** | Why "decode place" is not currently possible here: their *location* is a task dimension they built in; this repo presents one centred image per tick. |

### Yang, G.R., Joglekar, M.R., Song, H.F., Newsome, W.T., & Wang, X.-J. (2019). *Task representations in neural networks trained to perform many cognitive tasks.* Nature Neuroscience 22, 297–306.
`../Task representations in neural networks trained to perform many cognitive tasks.pdf`

- **256 recurrent units, single vanilla/tanh RNN** with fixation + 2 modality inputs + rule signal → ≈96k trainable parameters. This is the **primary parameter anchor**: the target for the redesigned recurrent core.
- The **rule/task-identity input vector** convention for multi-task RNNs — adopted for the multi-task diet arm, and deliberately *withheld* in the meta-RL arm.
- The 20-task cognitive suite (later "20-Cog-tasks").
- Functional clustering and mixed selectivity as the standard multi-task analysis.

### Khona, M., Chandra, S., Ma, J.J., & Fiete, I.R. (2023). *Winning the lottery with neural connectivity constraints: faster learning across cognitive tasks with spatially constrained sparse RNNs.* arXiv:2207.03523
`../2207.03523v2.pdf`

- The **locality-masked RNN (LM-RNN)** with sparsity as low as **4%** — the source of `models/gru_cell.py::make_locality_mask` and the S=1 worker.
- Crucially: "*by only training weights between nodes that correspond to edges on the graph, and setting all other weights to zero*" and "*these networks require far fewer parameters to train*". **This is what exposed audit finding A2**: the repo allocates the full dense `weight_hh` and masks at forward time, so its reported parameter count for S=1 is ~1.6× the number of weights actually being fit.
- "*LM-RNNs can perform as well or better than dense networks, when accounting for the **total number of nodes** or the **total number of synapses***" — the two parameter-matching conventions. The redesign matches on **effective synapses** as primary and adds one node-matched control cell, because you cannot satisfy both.
- **Mod-Cog** (up to 132 tasks) as an alternative multi-task battery to NeuroGym/Yang-19.

### Masse, N.Y., Yang, G.R., Song, H.F., Wang, X.-J., & Freedman, D.J. (2019). *Circuit mechanisms for the maintenance and manipulation of information in working memory.* Nature Neuroscience 22, 1159–1167. *(web only)*

- **100 recurrent units (80 excitatory / 20 inhibitory)** — the small end of the parameter anchor range, and the standard size for a single-WM-task neuro RNN.
- Activity-silent WM held in short-term synaptic efficacies — the motivation for arm **P** (Hebbian fast weights).

### Song, H.F., Yang, G.R., & Wang, X.-J. (2016). *Training excitatory-inhibitory recurrent neural networks for cognitive tasks.* PLoS Computational Biology 12(2): e1004792. *(web only)*

- The **80/20 E/I split** used by `model.dale_ei_split` — arm **D**.
- 100–150 unit networks; second parameter anchor.
- Alongside Yang et al. (2019) above: the natural anchor for the dense
  per-tick supervised (`SUP`) training regime and a near-unit recurrent-init
  spectral radius, used as the two isolated variables in `comments.txt`
  §13.4's vanilla init x supervision diagnostic (advisor.md §7b, 2026-08-01)
  — it decided which two confounds to test, not the diagnostic's outcome.

---

## 2. Behavioral gates and training criteria

- **Lei/Ito/Bashivan 2024** (above): >95% train, **>90% validation** — the primary gate anchor.
- **Molano-Mazón et al. / NeuroGym practice**: 90% accuracy as the criterion to stop training or advance to the next task in curriculum training of cognitive-task RNNs. *(web only)* Corroborates 0.90 as the field-standard threshold rather than the repo's ad hoc 0.95/0.80 pair.
- Human n-back accuracy profile (1-back ≈ 0.90–0.95, 2-back ≈ 0.80–0.85, 3-back ≈ 0.70) is the basis for the **graded inclusion gate** (0.90 / 0.85 / 0.80) the user proposed — kept as a *separate* gate from the convergence criterion, because "train until it converges" and "is this model behaviourally human-like" are different questions.

---

## 3. Representational geometry analyses (the new evaluation items)

### Libby, A., & Buschman, T.J. (2021). *Rotational dynamics reduce interference between sensory and memory representations.* Nature Neuroscience 24, 715–726. *(web only; code: github.com/buschman-lab/RotationalDynamics)*

The source for the **"rotations"** item the user asked to add:
- Sensory representations are **rotated into orthogonal memory representations** over time, which is what prevents new input from overwriting the stored item.
- The mechanism decomposes into **"stable" neurons** (selectivity preserved) and **"switching" neurons** (selectivity inverted) — a concrete, cheap per-unit analysis to run on `h_worker` / `h_flat`.
- Gives the prediction that the encode→maintain subspace angle should be near 90° in a well-trained model, and the falsifiable alternative (a purely persistent code would give 0°).

### Panichello, M.F., & Buschman, T.J. (2021). *Shared mechanisms underlie the control of working memory and attention.* Nature 593, 601–605. *(web only)*

The source for the **content vs context** framing:
- Selection acts by **dynamically transforming memories between subspaces** of neural activity, with prefrontal cortex as a domain-general controller.
- Operationalises "context" (which item / which position is currently relevant) as a separate, decodable subspace from "content" (what the item is) — the split adopted in the new `content_context.py` analysis.

### Lei/Ito/Bashivan 2024 (above) — the CTG and task-irrelevant-decoding methods.

---

## 4. Supervision, meta-RL, and "the network infers n"

### Wang, J.X., Kurth-Nelson, Z., Tirumala, D., Soyer, H., Leibo, J.Z., Munos, R., Blundell, C., Kumaran, D., & Botvinick, M. (2016). *Learning to reinforcement learn.* arXiv:1611.05763 — and Wang, J.X. et al. (2018). *Prefrontal cortex as a meta-reinforcement learning system.* Nature Neuroscience 21, 860–868. *(web only)*

- The **meta-RL recipe**: an RNN that receives the **previous reward and previous action as inputs**, trained by policy gradient over a *distribution* of related tasks, learns to implement a fast task-inference algorithm in its own recurrent dynamics with fixed weights.
- This is the correct formalisation of the user's "the RNN should find out the nature and n in the n-back task" — it is **task inference / meta-RL**, not unsupervised learning. Used verbatim for the `METARL` supervision arm: withhold the task cue, feed (r_{t-1}, a_{t-1}), block trials.
- The PFC-as-meta-RL framing also motivates the prediction that the **manager** (S=1) should carry the inferred task variable while the worker carries the memoranda.

---

## 5. Multi-task environments (the "general cognitive tasks" arm)

### Molano-Mazón, M., et al. (2022). *NeuroGym: An open resource for developing and sharing neuroscience tasks.* PsyArXiv. — package `neurogym` 2.3.1, PyPI. *(web only + wheel inspected locally)*

Chosen over hand-writing tasks. Verified present in the 2.3.1 wheel:
`bandit.py` (**one-armed / multi-armed bandit** — the user's request),
`dawtwostep.py` (**two-step / gambling**), `economicdecisionmaking.py`,
`delaymatchsample.py`, `dualdelaymatchsample.py`, `delaycomparison.py`,
`delaypairedassociation.py` (**working memory**), `gonogo.py`,
`contextdecisionmaking.py`, `perceptualdecisionmaking.py`,
`intervaldiscrimination.py`, `hierarchicalreasoning.py`, and
`collections/yang19.py` (**the Yang et al. 2019 20-task suite as a ready-made
collection**).

---

## 5b. Model size, tiny RNNs, and distillation

### Ji-An, L., Benna, M.K., & Mattar, M.G. (2025). *Discovering cognitive strategies with tiny recurrent neural networks.* Nature 644, 993.
`../s41586-025-09142-4.pdf`

- **1–4 unit RNNs (40–80 parameters)** outperform 30+ classical cognitive models at predicting individual animal and human choices. The evidence that very small recurrent models are scientifically serious, not toys.
- They used **GRUs** for the tiny networks ("other recurrent architectures are also applicable").
- Inputs are **(a_{t−1}, s_{t−1}, r_{t−1})** — the same previous-action/previous-reward convention as meta-RL, which is why the METARL arm and the tiny-RNN arm share an input path.
- All six tasks are **reward-learning / bandit family** (reversal, two-stage, transition-reversal two-stage, three-armed reversal, four-armed drifting bandit). This is the scope limit: none is a working-memory task, which is why tiny RNNs are specified for the bandit tasks in the multi-task diet but not as a model of image Sternberg.
- **Fig 2a is a teacher→student knowledge-distillation framework** — large teacher trained across subjects with a subject embedding, tiny students trained to match the teacher's output probabilities. Adopted directly as Phase 9.2.
- Their negative-log-likelihood vs *d* plots are **capacity curves**; adopted as Phase 9.1 to measure the effective dimensionality of the WM solution.

### Shakiba, M., Rokni, R., Mohammadi, M., & Dehghani, N. (2026). *Harnessing cortical geometry, wiring, and function as inductive biases for recurrent neural networks.* arXiv:2606.14975
`../shakiba26.pdf` — Neuromatch Academy + McGovern Institute, MIT.

312-unit SimpleRNN (one MICrONS field), Gaussian noise σ=0.05, Adam, 10 epochs, 20 runs, batch 128, on three tasks: One-Choice Inference, Perceptual Decision-Making, Go/NoGo. Eleven variants crossing biologically-informed weight init (W\*), permuted-value control (W!), real MICrONS coordinates (D\*) vs artificial grid (D), and communicability regularisation (C direct / C\* Earth-Mover's).

| Taken from it | Used for |
|---|---|
| **Four topology metrics** — entropy (Gaussian-KDE, 100 grid points), modularity Q (Clauset–Newman–Moore), small-worldness σ, **assortativity r** | Phase 8.9. Three already exist in `analysis/network_properties.py`; **assortativity does not and is being added**. Published target ranges: entropy 0.02–0.08 (structured) vs 3–6 (random-like); Q 0.4–0.5 vs 0.1; σ 1.5–2.5 vs ~1. |
| Their Fig 3 shows the four metrics **do not move together** (W\*D\*C\* reached low entropy without strong modularity or small-worldness) | Report all four separately; do not collapse into a "brain-likeness" score. |
| **Table 3, positive-only recurrence:** randomly-initialised variants collapse to **near-chance** (~0.25 / ~0.50); only functionally-initialised ones hold (W\*D\*C: 0.910/0.834/0.950) | **Corrects an error in an earlier draft of this project's spec.** A sign constraint is catastrophic *unless* the network starts from a bio-derived init. Every cell here uses uniform init, so the pre-registered prediction is that **arm D will cost substantial performance** — and a near-free D means the penalty is too weak, not that Dale's law is harmless. |
| **W! (permuted values, destroyed neuron-to-neuron assignment) performed comparably to W\*** | The benefit is in the **statistical structure of the weight distribution**, not the connectome mapping. So the new `M00001_bioinit` cell can use `LogNormal(-0.5, 0.5)`, rescaled to mean 0.1 / spectral radius 0.95, with **no MICrONS data at all**. The single most transferable result in the paper. |
| Their ECDF-resampling control *did* degrade | Use a fixed multiset of magnitudes, not fresh draws from a fitted marginal. |
| Functionally-initialised nets went **disassortative** (r < 0); spatially-constrained-but-randomly-initialised went **assortative** (r ≈ 0.4–0.5) | Why assortativity is worth adding: it dissociates two kinds of constraint the other three metrics conflate. |
| **Not adopted:** the MICrONS-derived mask itself | MICrONS is mouse visual cortex; this study aligns to human MTL/MFC. See `comments.txt` §9.5. |

### Yang et al. (2019) pretrained networks — `github.com/gyyang/multitask` *(web only)*
20 pretrained multi-task models on Google Drive, TensorFlow 1.8 / Python 2.7 era. Used as the external replication tier for the multi-task-vs-single-task comparison (Phase 10.1 Tier 2). Loaded by reading the TF1 checkpoint with TF2's `tf.train.load_checkpoint` and reimplementing the leaky-RNN forward pass in torch, rather than installing the original stack.

---

## 5c. Human behavioural gates

The single behavioural criterion is **not** taken from any paper. It is computed by `scripts/human_behavior_gates.py` from `response_accuracy` × `loads` in the DANDI NWB files themselves — the real Sternberg performance of the patients whose neurons are the alignment target (111 sessions, 15,259 trials) — and written to `results/human_behavior.csv`.

Literature values were used only as a sanity check on the resulting numbers:
- [BASHIVAN24]: ">95% on train, >90% on validation" — the practice of gating on a *generalization* set.
- Molano-Mazón et al. / PsychRNN practice: 90% as the field-standard stop criterion. *(web only)*

The derived criterion (000469, the only dataset covering all three loads) is **0.94 / 0.91 / 0.86** for loads 1/2/3 — below both literature values, because real patients under clinical conditions are not at ceiling.

---

## 6. Mechanisms already implemented (cited for completeness — no new decision taken)

| Reference | Implements |
|---|---|
| Miconi, T. (2017). *Biologically plausible learning in RNNs...* eLife 6:e20899 | Node perturbation with eligibility traces — `mechanisms/local_learning.py`, rungs 1–2 |
| Miconi, T., Clune, J., & Stanley, K. (2018). *Differentiable plasticity.* ICML | Hebbian fast weights `W_eff = W + α·Hebb` — arm **P**, `models/gru_cell.py::PlasticGRUCell` |
| Bellec, G., et al. (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons.* Nature Communications 11, 3625 | e-prop — local-learning rung 3 |
| Werfel, J., Xie, X., & Seung, H.S. (2005); Fiete, I.R., & Seung, H.S. (2006) | Node-perturbation gradient variance scales with unit count — the repo's stated reason the L-arm fails. **Audit finding A5 argues this excuse does not hold at the observed chance-level performance.** |
| O'Reilly, R.C., & Frank, M.J. (2006). *Making working memory work.* Neural Computation 18, 283–328 | PBWM 3-gate manager — `models/gru_cell.py::PBWMManagerCell` |
| Schuessler, F., et al. (2024) | Aligned vs oblique readout regime — `model.readout_scale` |
| Frankle, J., & Carbin, M. (2019). *The lottery ticket hypothesis.* ICLR. `../1803.03635v5.pdf` | The "fixed spatial lottery ticket at init" framing of the S=1 mask (via Khona et al.) |

## 7. Neural datasets

| Dataset | Reference | Used for |
|---|---|---|
| DANDI 000469, 000673 | Kamiński / Rutishauser human MTL+MFC picture Sternberg | Tier A alignment target |
| DANDI 001187 | Daume, J., et al. (2024) | Tier B replication; the **only** source paper giving explicit unit QC numbers (`min_firing_hz: 0.1`, `min_session_accuracy: 0.55`), applied uniformly across datasets |
| DANDI 000574 | verbal WM, different schema | Tier C — needs its own adapter, not built |

---

## Not used (read, judged not decision-relevant here)

`../Efficiently Modeling Long Sequences with Structured State Spaces.pdf` (S4),
`../RIM.pdf` (Recurrent Independent Mechanisms), `../ReflexGrad.pdf`,
`../chatgpt_wm.pdf`, `../Mioni, 2017.pdf` — interesting adjacent architectures,
but adopting any of them would replace the GRU substrate the whole 5-arm
battery is defined on. Noted here so a future session does not re-read them
expecting a pending decision.
