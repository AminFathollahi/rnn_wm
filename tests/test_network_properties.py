import numpy as np
import pandas as pd

from brainalign_wm.analysis.network_properties import (
    degree_assortativity,
    gru_effective_connectivity,
    mixed_selectivity_index,
    modularity_q,
    small_worldness,
    weight_entropy,
)


def test_weight_entropy_point_mass_is_zero():
    W = np.zeros((10, 10))
    assert weight_entropy(W) == 0.0


def test_weight_entropy_uniform_is_near_max():
    rng = np.random.RandomState(0)
    W = rng.uniform(-1, 1, size=(1000, 1000))
    h = weight_entropy(W, n_bins=50)
    assert h > np.log2(50) * 0.8  # close to max entropy for a near-uniform distribution


def test_weight_entropy_kde_point_mass_is_near_zero():
    W = np.zeros((10, 10))
    assert weight_entropy(W, method="kde") == 0.0  # short-circuits before the KDE path (max==min)


def test_weight_entropy_kde_uniform_exceeds_concentrated():
    """Item 8.9a: KDE entropy has a different scale than the histogram
    method's `log2(n_bins)` ceiling, so this uses a relative comparison
    instead -- a flat (uniform) density must have higher KDE entropy than
    a peaked (Gaussian) one over the SAME grid resolution. NOTE: the gap
    is modest (empirically ~0.6-0.75 bits here), not dramatic, because
    Scott's-rule KDE bandwidth and the [min,max] grid both scale with the
    sample's own spread -- so this method is largely SCALE-invariant
    (tightening/widening a Gaussian by 10x barely moves its entropy, see
    git history/PHASE_LOG.md) and mostly sensitive to distribution SHAPE."""
    rng = np.random.RandomState(0)
    uniform_vals = rng.uniform(-1, 1, size=2000)
    concentrated_vals = rng.randn(2000) * 0.01
    h_uniform = weight_entropy(uniform_vals, method="kde", n_grid=100)
    h_concentrated = weight_entropy(concentrated_vals, method="kde", n_grid=100)
    print("KDE entropy uniform:", h_uniform, "concentrated:", h_concentrated)
    assert h_uniform > h_concentrated + 0.3


def test_weight_entropy_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        weight_entropy(np.ones((5, 5)) * np.arange(5), method="bogus")


def test_gru_effective_connectivity_shape_and_values():
    H = 4
    weight_hh = np.zeros((3 * H, H))
    weight_hh[0, 1] = 2.0  # gate 0, unit 0 <- unit 1
    weight_hh[H + 0, 1] = -3.0  # gate 1, unit 0 <- unit 1
    C = gru_effective_connectivity(weight_hh)
    assert C.shape == (H, H)
    assert np.isclose(C[0, 1], 5.0)  # |2| + |-3|
    assert np.all(np.diag(C) == 0.0)


def test_gru_effective_connectivity_rejects_bad_shape():
    import pytest

    with pytest.raises(ValueError):
        gru_effective_connectivity(np.zeros((10, 4)))  # 10 not a multiple of 4


def test_modularity_q_detects_block_structure():
    # Two disconnected 5-node cliques -> Louvain should find exactly that
    # partition, giving a high (near-maximal) modularity.
    n = 10
    C = np.zeros((n, n))
    C[:5, :5] = 1.0
    C[5:, 5:] = 1.0
    np.fill_diagonal(C, 0.0)
    q = modularity_q(C, seed=0)
    assert q is not None
    assert q > 0.3  # two clean disconnected blocks -> strongly positive Q


def test_modularity_q_empty_graph_returns_none():
    assert modularity_q(np.zeros((5, 5))) is None


def test_modularity_q_cnm_detects_block_structure():
    """Item 8.9c: same two-clique planted structure as the Louvain test
    above, with `method="cnm"` (Clauset-Newman-Moore greedy modularity,
    [SHAKIBA26]'s own estimator) -- CNM should also find a high-Q partition
    on such a clean case, even though it isn't guaranteed to match Louvain
    exactly on harder graphs."""
    n = 10
    C = np.zeros((n, n))
    C[:5, :5] = 1.0
    C[5:, 5:] = 1.0
    np.fill_diagonal(C, 0.0)
    q = modularity_q(C, method="cnm")
    assert q is not None
    assert q > 0.3


def test_modularity_q_unknown_method_raises():
    import pytest

    n = 10
    C = np.zeros((n, n))
    C[:5, :5] = 1.0
    C[5:, 5:] = 1.0
    with pytest.raises(ValueError):
        modularity_q(C, method="bogus")


def test_small_worldness_detects_small_world_structure():
    """Audit fix T1: the old version of this test only asserted `result is
    None or isinstance(result, float)`, which is trivially true of every
    possible return value and so never actually exercised the metric. A
    Watts-Strogatz ring-lattice-plus-slight-rewiring graph (high
    clustering, short path length -- the textbook small-world regime) must
    give sigma > 1.0; a fully random graph would not."""
    import networkx as nx

    n = 30
    G = nx.watts_strogatz_graph(n, k=4, p=0.1, seed=0)
    C = nx.to_numpy_array(G)
    sigma = small_worldness(C, density=0.3, n_random=2, n_iter=2, seed=0, max_nodes=n)
    assert sigma is not None
    assert sigma > 1.0


def test_small_worldness_returns_float_or_none_on_random_graph():
    rng = np.random.RandomState(0)
    n = 30
    C = rng.uniform(0, 1, size=(n, n))
    np.fill_diagonal(C, 0.0)
    result = small_worldness(C, density=0.2, n_random=2, n_iter=2, seed=0)
    assert result is None or isinstance(result, float)


def test_small_worldness_too_small_component_returns_none():
    C = np.zeros((5, 5))
    C[0, 1] = C[1, 0] = 1.0
    assert small_worldness(C, density=0.5) is None


# ---------------- 8.9d: degree assortativity (NEW, not previously implemented) ----------------

def test_degree_assortativity_positive_for_same_degree_clusters():
    """Two fully-connected cliques of DIFFERENT sizes (so different
    within-clique degree: ~2 in the small one, ~7 in the large one),
    joined by a SINGLE bridge edge. Nearly every edge pairs two nodes of
    similar degree (within-clique), with only one edge crossing the
    degree gap -- classic assortative structure, r should be strongly
    positive. (An earlier draft of this test tried a hub-clique-plus-
    leaves construction expecting positive r "because hubs connect to
    hubs" -- that construction actually measured r < 0: a majority of
    same-degree edges is not sufficient, degree ASSORTATIVITY is a
    correlation over edge endpoints, and hub-leaf edges pull it down hard
    when leaves visit only hubs. This two-clique-plus-bridge construction
    verified at r ~ 0.77.)"""
    import networkx as nx

    G = nx.Graph()
    small = list(range(3))
    large = list(range(10, 18))
    G.add_edges_from((i, j) for i in small for j in small if i < j)
    G.add_edges_from((i, j) for i in large for j in large if i < j)
    G.add_edge(small[0], large[0])  # the only cross-degree edge
    C = nx.to_numpy_array(G)
    r = degree_assortativity(C, density=0.9, max_nodes=C.shape[0])
    print("degree_assortativity (two cliques + bridge):", r)
    assert r is not None
    assert r > 0.5, r


def test_degree_assortativity_negative_for_star_structure():
    """A handful of star graphs (one central hub connected only to many
    low-degree leaves, leaves never connected to each other) glued at
    their hubs into one connected component -- classic hub-periphery,
    disassortative structure: r should be clearly negative."""
    import networkx as nx

    G = nx.Graph()
    hub_ids = [0, 1, 2]
    leaf_id = 10
    for hub in hub_ids:
        for _ in range(6):
            G.add_edge(hub, leaf_id)
            leaf_id += 1
    for i in range(len(hub_ids) - 1):
        G.add_edge(hub_ids[i], hub_ids[i + 1])  # connect the stars' hubs so the graph is one component
    C = nx.to_numpy_array(G)
    r = degree_assortativity(C, density=0.9, max_nodes=C.shape[0])
    print("degree_assortativity (star-of-stars):", r)
    assert r is not None
    assert r < -0.2, r


def test_degree_assortativity_too_small_component_returns_none():
    C = np.zeros((5, 5))
    C[0, 1] = C[1, 0] = 1.0
    assert degree_assortativity(C, density=0.5) is None


def _synthetic_activity_df(n_units=6, seed=0) -> pd.DataFrame:
    """2 loads x 2 items, 6 trials/cell, deterministic per-cell mean plus
    noise, with a large, controlled interaction term on unit 0 only."""
    rng = np.random.RandomState(seed)
    rows = []
    cell_bases = {
        (1, 10): np.zeros(n_units),
        (1, 20): np.zeros(n_units),
        (2, 10): np.zeros(n_units),
        (2, 20): np.zeros(n_units),
    }
    # Unit 0: pure interaction (XOR-like) -- only (1,10) and (2,20) are high.
    cell_bases[(1, 10)][0] = 5.0
    cell_bases[(2, 20)][0] = 5.0
    # Units 1-5: pure additive load main effect only (no item effect, no
    # interaction) -- a real, non-degenerate main effect large enough to
    # dominate the injected noise, so the interaction ratio is legitimately
    # near 0 rather than ill-defined (unit 0's own noiseless-off cells make
    # this distinction matter: total_ss must not be ~0 for these units).
    for (load, item) in cell_bases:
        cell_bases[(load, item)][1:] = 5.0 if load == 1 else -5.0
    trial_id = 0
    for (load, item), base in cell_bases.items():
        for _ in range(6):
            for t in range(3):  # 3 ticks per trial, all epoch="maintain"
                h = base + rng.normal(0, 0.01, size=n_units)
                rows.append({
                    "trial_id": trial_id, "epoch": "maintain", "load": load,
                    "held_items": [item], "h_flat": h.tolist(), "h_worker": None, "h_manager": None,
                })
            trial_id += 1
    return pd.DataFrame(rows)


def test_mixed_selectivity_index_detects_interaction_unit():
    df = _synthetic_activity_df()
    result = mixed_selectivity_index(df, epoch="maintain", min_trials_per_cell=2)
    assert result is not None
    assert result["unit_index"][0] > 0.5  # unit 0 has a strong, controlled XOR-like interaction
    assert np.all(result["unit_index"][1:] < 0.1)  # every other unit is flat (no signal at all)


def test_mixed_selectivity_index_none_when_too_few_cells():
    df = _synthetic_activity_df()
    df = df[df["load"] == 1]  # only one load level left -> no interaction term possible
    assert mixed_selectivity_index(df, epoch="maintain") is None
