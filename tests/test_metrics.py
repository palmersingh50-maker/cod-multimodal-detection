import numpy as np
import pytest

from cod.evaluation.metrics import CODMetrics, e_measure, iou_dice, mae, s_measure, weighted_f


def rect(h=64, w=80):
    g = np.zeros((h, w))
    g[16:48, 20:60] = 1
    return g


def test_perfect_prediction():
    g = rect()
    assert mae(g, g) == 0
    assert s_measure(g, g) == pytest.approx(1, abs=1e-6)
    assert weighted_f(g, g) == pytest.approx(1, abs=1e-6)
    # mean over 256 thresholds; threshold 0 marks every pixel foreground, so a perfect binary map scores 255/256+
    assert e_measure(g, g) == pytest.approx(1, abs=5e-3)
    assert iou_dice(g, g) == (1.0, 1.0)


def test_inverted_prediction_is_bad():
    g = rect()
    p = 1 - g
    assert mae(p, g) == 1
    assert s_measure(p, g) < 0.1 and weighted_f(p, g) < 0.05 and e_measure(p, g) < 0.1
    assert iou_dice(p, g) == (0.0, 0.0)


def test_monotone_with_noise():
    g = rect()
    rng = np.random.default_rng(0)
    clean = np.clip(g * 0.9 + 0.05, 0, 1)
    noisy = np.clip(clean + rng.normal(0, 0.3, g.shape), 0, 1)
    assert s_measure(clean, g) > s_measure(noisy, g)
    assert weighted_f(clean, g) > weighted_f(noisy, g)


def test_empty_gt_edge_cases():
    g = np.zeros((32, 32))
    assert s_measure(g, g) == 1.0 and e_measure(g, g) > 0.99 and weighted_f(g, g) == 1.0
    assert s_measure(np.ones_like(g), g) == 0.0


def test_accumulator_means():
    m = CODMetrics()
    g = rect()
    m.update(g, g), m.update(1 - g, g)
    r = m.compute()
    assert r["n"] == 2 and r["mae"] == pytest.approx(0.5) and r["miou"] == pytest.approx(0.5)


def test_against_pysodmetrics_reference():
    sod = pytest.importorskip("py_sod_metrics")
    g = rect()
    rng = np.random.default_rng(1)
    p = np.clip(g * 0.7 + 0.1 + rng.normal(0, 0.1, g.shape), 0, 1)
    p8, g8 = (p * 255).round().astype(np.uint8), (g * 255).astype(np.uint8)
    S, W, E, A = sod.Smeasure(), sod.WeightedFmeasure(), sod.Emeasure(), sod.MAE()
    for o in (S, W, E, A):
        o.step(p8, g8)
    assert s_measure(p8 / 255.0, g) == pytest.approx(S.get_results()["sm"], abs=2e-3)
    assert weighted_f(p8 / 255.0, g) == pytest.approx(W.get_results()["wfm"], abs=1e-3)
    assert e_measure(p8 / 255.0, g) == pytest.approx(E.get_results()["em"]["curve"].mean(), abs=1e-3)
    assert mae(p8 / 255.0, g) == pytest.approx(A.get_results()["mae"], abs=1e-4)
