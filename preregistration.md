# Pre-registration (freeze before the full training grid)

Copy the directional predictions + decision table from protocol §2 here, commit, and
record the commit hash. Do not edit after the full grid starts.

## Primary hypotheses (fill in / confirm from §2)
- H1 (Structure -> anatomy): S=1 raises alignment; worker↔MTL, manager↔MFC dissociation.
- H2 (Modulation -> frontal control): M=1 raises MFC-epoch alignment; reflection-shuffle causal.
- H3 (Learning): L=1 ≥ BPTT alignment at matched behavior, *conditional on the gate*.
- H4 (Target): M111 best; S×M interaction > 0.
- H5 (Oblique/dynamic): oblique regime more brain-aligned; stable+dynamic delay code.
- H6 (Persistent activity): constrained models show more memoranda-selective persistence.

## Primary region per hypothesis (avoid region-shopping)
- H1-worker / H6: MTL.   H2: MFC.

## Analysis lock
- Metric: crossnobis RSA, noise-ceiling-normalized; inference by condition-label permutation.
- Model: align ~ S*M*L + accuracy + (1|seed) + (1|patient); FDR across regions.
- Seeds: >= 8. Noise ceiling: trial split-half / LOSO.

Frozen at commit: __________  (date: __________)
