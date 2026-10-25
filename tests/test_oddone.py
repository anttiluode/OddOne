import numpy as np

from oddone import LH, LW, LinearLens, Score, count_channels, panel, split, unvec, vec
from sim_occluder import count_above, transport

D = LW * LH * 3


def hidden_stream(rng, n):
    """Low-res hidden-half vectors: a dim background with a moving bright blob."""
    yy, xx = np.mgrid[0:LH, 0:LW]
    out = []
    for t in range(n):
        cy, cx = rng.uniform(0, LH), rng.uniform(0, LW)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)
        img = 0.2 + 0.6 * blob[..., None] * np.array([1.0, 0.8, 0.6])
        out.append(img.ravel())
    return np.array(out)


def run(lens, Y, X):
    s = Score(rate=0.05)
    for y, x in zip(Y, X):
        pred = lens.predict(y)
        _, mx = lens.means()
        if lens.n > 50:
            s.add(float(np.sum((x - pred) ** 2)), float(np.sum((x - mx) ** 2)))
        lens.learn(y, x)
    return s.value()


def test_linear_lens_reads_a_linear_leak():
    rng = np.random.default_rng(0)
    X = hidden_stream(rng, 1500)
    G = rng.normal(size=(D, D)) / np.sqrt(D) * 0.3          # light transport, hidden -> visible
    Y = 0.4 + X @ G.T + 0.01 * rng.normal(size=X.shape)
    assert run(LinearLens(D, D, background=False), Y, X) > 0.8


def test_linear_lens_does_not_hallucinate_from_unrelated_input():
    rng = np.random.default_rng(1)
    X = hidden_stream(rng, 1500)
    Y = 0.4 + 0.1 * rng.normal(size=X.shape)                  # visible half carries nothing
    assert run(LinearLens(D, D, background=False), Y, X) < 0.1


def test_channels_count_matches_a_low_rank_leak():
    rng = np.random.default_rng(2)
    n = 2000
    X = hidden_stream(rng, n)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    leak = Xc @ Vt[:3].T                                      # 3 hidden patterns leak through
    Y = leak @ rng.normal(size=(3, D)) + 0.05 * rng.normal(size=X.shape)
    c = count_channels(Y, X)
    assert (c > 0.5).sum() == 3


def test_split_and_resize_shapes():
    frame = np.random.default_rng(3).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    vis, hid = split(frame)
    assert vis.shape == hid.shape == (480, 320, 3)
    v = vec(hid)
    assert v.shape == (D,)
    assert unvec(v, hid.shape[:2]).shape == hid.shape
    assert panel(hid, hid, ["x"], np.array([0.9, 0.2]), 640).shape[1] == 640


def test_edge_passes_far_more_than_a_bare_wall():
    bare = np.linalg.svd(transport(occluder=False), compute_uv=False)
    edge = np.linalg.svd(transport(occluder=True), compute_uv=False)
    assert count_above(edge, 1e-2) > 10 * count_above(bare, 1e-2)
