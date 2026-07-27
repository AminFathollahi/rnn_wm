"""Internal, brain-like organization metrics computed directly from a
trained model's own weights/activity -- no neural recordings involved
(unlike every other module under `analysis/`, which compares a model
against real Tier-A single-neuron data). These are independent evidence:
if biological constraints (S/M/P) also produce more brain-like *internal*
organization, that corroborates the neural-alignment DVs without sharing
any of their session-count/noise-ceiling limitations.

Cross-temporal decoding stability (H5) already exists in
`dynamics_and_persistence.py::stability_index_for_session` and is not
duplicated here.

    - `weight_entropy`: Shannon entropy (bits) of a weight tensor's value
      distribution -- higher entropy roughly tracks a more uniformly-used
      weight range; lower entropy tracks a more concentrated/sparse one.
    - `gru_effective_connectivity`: collapses a GRU-family cell's
      `weight_hh` [n_gates*H, H] (3 gates for `MaskedGRUCell`/
      `PlasticGRUCell`/`nn.GRUCell`, 4 for `PBWMManagerCell`) into a single
      [H, H] "how much does unit j drive unit i's update, summed across
      gates" connectivity matrix -- the natural undirected-graph input for
      `modularity_q`/`small_worldness` below.
    - `modularity_q`: Louvain community modularity Q on that connectivity
      graph.
    - `small_worldness`: Watts-Strogatz sigma (clustering/path-length
      ratio vs. degree-matched random graphs) on a thresholded, undirected
      version of the same graph.
    - `mixed_selectivity_index`: per-unit fraction of (load x held-item)
      cell-mean variance attributable to the load-by-item INTERACTION term
      (Rigotti et al. 2013's nonlinear mixed selectivity), computed from a
      replayed activity log. Held-item identity, not category, is used as
      the second factor: the real-data replay path
      (`generate_activity_logs.generate_activity_log`) never populates
      `held_categories`/`probe_category` (hardcoded to `""` -- the DANDI
      item-identity mapping carries no semantic category label), but
      `held_items` (session-local item index) is always populated and is a
      legitimate, non-post-hoc delay-period variable in its own right.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


def weight_entropy(W: np.ndarray, n_bins: int = 50, method: str = "histogram", n_grid: int = 100) -> float:
    """Shannon entropy (bits) of `W`'s value distribution (Sheeran et al.
    2024's weight-entropy diagnostic).

    `method="histogram"` (default, kept for backward compatibility):
    `W` flattened and binned over its own [min, max] range into `n_bins`
    bins; a completely uniform histogram gives log2(n_bins) (max entropy),
    a point mass gives 0.

    `method="kde"` (item 8.9a, comments.txt §5): [SHAKIBA26] use a Gaussian
    KDE over `n_grid=100` grid points spanning `W`'s value range instead of
    a histogram, so their published entropy regimes (high ~3-6, random-
    like; intermediate ~0.6-1.5; low ~0.02-0.6 -- verified verbatim against
    `../shakiba26.pdf` p.4 body text, "entropy variants (W and WD*C;
    entropy ~3-6)... Intermediate-entropy variants... entropy ~0.6-1.5...
    Low-entropy variants... entropy ~0.02-0.6"; comments.txt's own
    paraphrase said "~0.02-0.08", the real range is wider) are directly
    comparable to numbers from this function.
    `scipy.stats.gaussian_kde` estimates a continuous density over an
    `n_grid`-point grid; that grid IS the discretization the entropy sum is
    taken over (each grid point treated as one bin of width
    `(max-min)/(n_grid-1)`, density values renormalized to sum to 1 over
    the grid before the Shannon-entropy formula -- a KDE has no natural
    "bin" without an explicit grid, this is the natural analogue of the
    histogram method's bin count)."""
    vals = np.asarray(W).ravel()
    if vals.size == 0 or np.allclose(vals.max(), vals.min()):
        return 0.0
    if method == "histogram":
        counts, _ = np.histogram(vals, bins=n_bins)
        p = counts / counts.sum()
    elif method == "kde":
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(vals)
        grid = np.linspace(vals.min(), vals.max(), n_grid)
        density = kde(grid)
        p = density / density.sum()
    else:
        raise ValueError(f"weight_entropy: unknown method {method!r}, expected 'histogram' or 'kde'")
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gru_effective_connectivity(weight_hh: np.ndarray) -> np.ndarray:
    """[n_gates*H, H] -> [H, H], summed |gate block| per (i,j), diagonal
    zeroed (self-connections aren't a graph edge for modularity/small-
    world purposes)."""
    weight_hh = np.asarray(weight_hh)
    n_out, H = weight_hh.shape
    if n_out % H != 0:
        raise ValueError(f"weight_hh shape {weight_hh.shape} not a multiple of hidden_dim={H}")
    n_gates = n_out // H
    C = np.abs(weight_hh.reshape(n_gates, H, H)).sum(axis=0)
    np.fill_diagonal(C, 0.0)
    return C


def modularity_q(
    C: np.ndarray, seed: int = 0, resolution: float = 1.0, method: str = "louvain",
) -> Optional[float]:
    """Community modularity Q of the weighted undirected graph built from
    connectivity matrix `C` (`C[i,j]` = edge weight, symmetrized via C+C.T
    since GRU connectivity is inherently directed but modularity on a
    signed-direction graph isn't well-defined). Returns None if the graph
    is empty/disconnected-only (fewer than 2 nodes with any edge).

    `method="louvain"` (default, kept for backward compatibility):
    `nx.algorithms.community.louvain_communities`.

    `method="cnm"` (item 8.9b, comments.txt §5): Clauset-Newman-Moore
    GREEDY modularity maximization, [SHAKIBA26]'s own estimator (their Q
    ranges -- strongly modular ~0.4-0.5, weakly constrained ~0.1 --
    verified against `../shakiba26.pdf` p.4, "developed the strongest
    community structure (Q ~ 0.4-0.5)... remained only weakly modular
    (Q ~ 0.1)" -- were measured with CNM, not Louvain; the two are
    different algorithms with no guaranteed-identical output on the same
    graph, so report which one produced a given number, not just
    "modularity Q")."""
    C = np.asarray(C)
    C_sym = C + C.T
    G = nx.from_numpy_array(C_sym)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_edges() == 0:
        return None
    if method == "louvain":
        communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    elif method == "cnm":
        communities = nx.algorithms.community.greedy_modularity_communities(
            G, weight="weight", resolution=resolution
        )
    else:
        raise ValueError(f"modularity_q: unknown method {method!r}, expected 'louvain' or 'cnm'")
    return float(nx.algorithms.community.modularity(G, communities, weight="weight", resolution=resolution))


def _thresholded_largest_cc_graph(C: np.ndarray, density: float, max_nodes: int, seed: int) -> Optional[nx.Graph]:
    """Shared graph construction for `small_worldness`/`degree_assortativity`
    (item 8.9d factored this out of `small_worldness`, which had it
    inline, so both metrics see the SAME graph): subsample to `max_nodes`
    (fixed `seed`, reproducible), symmetrize, keep the top `density`
    fraction of edges by magnitude, return the largest connected component
    (both `sigma` and `degree_assortativity_coefficient` want a connected
    graph). Returns `None` if the result would be empty/too small (<10
    nodes) for either metric to be meaningful."""
    C = np.asarray(C)
    if C.shape[0] > max_nodes:
        idx = np.random.RandomState(seed).choice(C.shape[0], size=max_nodes, replace=False)
        C = C[np.ix_(idx, idx)]
    C_sym = C + C.T
    n = C_sym.shape[0]
    flat = C_sym[np.triu_indices(n, k=1)]
    if flat.size == 0 or np.count_nonzero(flat) == 0:
        return None
    thresh = np.quantile(flat[flat > 0], 1.0 - density) if np.any(flat > 0) else np.inf
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in zip(*np.triu_indices(n, k=1)):
        if C_sym[i, j] >= thresh:
            G.add_edge(int(i), int(j))
    if G.number_of_nodes() == 0:
        return None
    largest_cc = max(nx.connected_components(G), key=len)
    if len(largest_cc) < 10:
        return None
    return G.subgraph(largest_cc).copy()


def small_worldness(
    C: np.ndarray, density: float = 0.1, n_random: int = 2, n_iter: int = 2, seed: int = 0, max_nodes: int = 64,
) -> Optional[float]:
    """Watts-Strogatz sigma on an unweighted graph kept to the top
    `density` fraction of `C`'s (symmetrized) edges by magnitude --
    computing sigma directly on a dense weighted graph isn't standard and
    `nx.algorithms.smallworld.sigma`'s random-reference rewiring cost grows
    steeply with edge count (measured: ~0.3s at 50 nodes/~120 edges, ~9s at
    120 nodes/~700 edges -- worker/flat cores at their native 196-512
    units would take minutes PER graph, and this runs once per component
    per completed checkpoint). `C` is randomly subsampled to `max_nodes`
    units first (fixed `seed`, so reproducible) to keep this tractable
    across a whole grid's worth of checkpoints; `n_random`/`n_iter` are
    also deliberately small (a coarse, fast estimate, not a converged one
    -- same tradeoff `rsa.py::within_session_noise_ceiling` makes for its
    own resample count) -- increase locally (and/or raise `max_nodes`) if
    a tighter estimate is needed for one specific cell. Computed on the
    largest connected component of the subsampled graph only (`sigma`
    requires a connected graph); returns None if that component is too
    small (<10 nodes) for the random-reference comparison to be
    meaningful."""
    G_cc = _thresholded_largest_cc_graph(C, density, max_nodes, seed)
    if G_cc is None:
        return None
    try:
        return float(nx.algorithms.smallworld.sigma(G_cc, niter=n_iter, nrand=n_random, seed=seed))
    except (nx.NetworkXError, ZeroDivisionError):
        return None


def degree_assortativity(C: np.ndarray, density: float = 0.1, seed: int = 0, max_nodes: int = 64) -> Optional[float]:
    """Item 8.9d (comments.txt §5, NOT previously implemented): degree
    correlation `r` of the SAME thresholded/subsampled graph
    `small_worldness` builds (`_thresholded_largest_cc_graph`, ~3 lines on
    top of `gru_effective_connectivity`, per the spec's own estimate).
    `r > 0`: hubs preferentially connect to other hubs (hub-rich
    integrative structure). `r < 0`: hub-periphery structure. [SHAKIBA26]:
    spatially-constrained-but-randomly-initialized variants went
    positively assortative (r ~ 0.4-0.5); functionally initialized ones
    went DISASSORTATIVE (r < 0) -- verified against `../shakiba26.pdf` p.4,
    "WD*C... positive assortativity across tasks (r ~ 0.4-0.5)... [W*D*C
    and W!D*C*] were disassortative (r < 0)" and Fig. 3c (p.7) -- this
    metric dissociates two kinds of constraint the other three topology
    metrics (entropy/modularity/small-worldness) conflate (item 8.9's own
    "these four do NOT move together" warning, also Fig. 3's own caption,
    p.7)."""
    G_cc = _thresholded_largest_cc_graph(C, density, max_nodes, seed)
    if G_cc is None:
        return None
    try:
        r = nx.degree_assortativity_coefficient(G_cc)
    except (nx.NetworkXError, ZeroDivisionError):
        return None
    return float(r) if r == r else None  # nan-check without importing math for one comparison


def _epoch_trial_means(df: pd.DataFrame, epoch: str) -> pd.DataFrame:
    """One row per trial: (load, item, unit activity vector), averaged over
    that trial's ticks in `epoch`. `item` = the first held item's
    session-local identity index (see module docstring for why identity,
    not category, is used)."""
    sub = df[df.epoch == epoch]
    if len(sub) == 0:
        return pd.DataFrame()
    is_flat = sub["h_flat"].iloc[0] is not None
    rows = []
    for trial_id, g in sub.groupby("trial_id"):
        held = g["held_items"].iloc[0]
        if held is None or len(held) == 0:
            continue
        if is_flat:
            h = np.stack(g["h_flat"].to_numpy()).mean(axis=0)
        else:
            h = np.concatenate(
                [np.stack(g["h_worker"].to_numpy()).mean(axis=0), np.stack(g["h_manager"].to_numpy()).mean(axis=0)]
            )
        rows.append({"trial_id": trial_id, "load": int(g["load"].iloc[0]), "item": int(held[0]), "h": h})
    return pd.DataFrame(rows)


def mixed_selectivity_index(
    df: pd.DataFrame, epoch: str = "maintain", min_trials_per_cell: int = 2,
) -> Optional[dict]:
    """Per-unit nonlinear mixed-selectivity index (Rigotti et al. 2013):
    for each unit, decompose its (load x item) cell-mean activity into
    grand mean + main effects (load, item) + interaction, via the standard
    ANOVA sum-of-squares split (weighted by cell trial count, so unequal
    cell sizes -- unavoidable here since not every item recurs at every
    load -- don't bias the estimate). `index = interaction_SS /
    total_SS`, clipped to [0,1]; 0 = purely additive (linearly separable)
    selectivity, 1 = selectivity fully explained by load-item conjunctions.
    `df` must already be filtered to a single session (`trial_id` is only
    session-locally unique -- see `dynamics_and_persistence.py`'s same
    caveat); item identity is itself session-local, so pooling across
    sessions before this decomposition would conflate unrelated items.

    Returns {"unit_index": array[n_units], "population_mean": float,
    "n_cells": int} or None if too few (load, item) cells have
    >=min_trials_per_cell trials to estimate an interaction term at all."""
    trial_means = _epoch_trial_means(df, epoch)
    if trial_means.empty:
        return None
    cell_counts = trial_means.groupby(["load", "item"]).size()
    valid_cells = set(cell_counts[cell_counts >= min_trials_per_cell].index)
    trial_means = trial_means[trial_means.apply(lambda r: (r["load"], r["item"]) in valid_cells, axis=1)]
    if len(valid_cells) < 4 or trial_means.empty:  # need >=2 loads x >=2 items to have an interaction term at all
        return None
    loads = sorted(trial_means["load"].unique())
    items = sorted(trial_means["item"].unique())
    if len(loads) < 2 or len(items) < 2:
        return None

    H = np.stack(trial_means["h"].to_numpy())  # [n_trials, n_units]
    n_units = H.shape[1]
    grand_mean = H.mean(axis=0)  # [n_units]

    cell_mean = {}
    cell_n = {}
    for (l, it), g in trial_means.groupby(["load", "item"]):
        if (l, it) not in valid_cells:
            continue
        cell_mean[(l, it)] = np.stack(g["h"].to_numpy()).mean(axis=0)
        cell_n[(l, it)] = len(g)

    total_n = sum(cell_n.values())
    load_mean = {
        l: sum(cell_n[(l, it)] * cell_mean[(l, it)] for it in items if (l, it) in cell_mean)
        / sum(cell_n[(l, it)] for it in items if (l, it) in cell_mean)
        for l in loads if any((l, it) in cell_mean for it in items)
    }
    item_mean = {
        it: sum(cell_n[(l, it)] * cell_mean[(l, it)] for l in loads if (l, it) in cell_mean)
        / sum(cell_n[(l, it)] for l in loads if (l, it) in cell_mean)
        for it in items if any((l, it) in cell_mean for l in loads)
    }

    total_ss = np.zeros(n_units)
    interaction_ss = np.zeros(n_units)
    for (l, it), mu_cell in cell_mean.items():
        n = cell_n[(l, it)]
        total_ss += n * (mu_cell - grand_mean) ** 2
        interaction_term = mu_cell - load_mean[l] - item_mean[it] + grand_mean
        interaction_ss += n * interaction_term ** 2

    with np.errstate(invalid="ignore", divide="ignore"):
        unit_index = np.clip(np.nan_to_num(interaction_ss / total_ss, nan=0.0), 0.0, 1.0)
    return {
        "unit_index": unit_index,
        "population_mean": float(unit_index.mean()),
        "n_cells": len(valid_cells),
        "n_trials": int(total_n),
    }
