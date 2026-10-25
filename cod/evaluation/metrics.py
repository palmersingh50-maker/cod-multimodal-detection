"""COD evaluation metrics (numpy, per image, predictions & GT at the same resolution).

pred: float array in [0,1]; gt: bool/{0,1} array. Implementations follow
  S-measure   Fan et al., ICCV 2017
  weighted-F  Margolin et al., CVPR 2014
  E-measure   Fan et al., IJCAI 2018 (mean over 256 thresholds)
and are cross-checked against the PySODMetrics reference in tests/ (when installed).
mIoU / Dice: pred binarised at `threshold`; foreground IoU/Dice per image, averaged over images.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist

EPS = np.spacing(1)


def mae(pred, gt):
    return float(np.abs(pred - gt).mean())


# ---------------------------------------------------------------- S-measure
def _obj_score(pred, gt):
    x = pred[gt == 1]
    if x.size == 0:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1) if x.size > 1 else 0.0
    return float(2 * m / (m * m + 1 + s + EPS))


def _s_object(pred, gt):
    u = gt.mean()
    return u * _obj_score(pred, gt) + (1 - u) * _obj_score(1 - pred, 1 - gt)


def _centroid(gt):
    h, w = gt.shape
    if gt.sum() == 0:
        return h // 2, w // 2
    ys, xs = np.nonzero(gt)
    return int(round(ys.mean())), int(round(xs.mean()))


def _ssim(p, g):
    n = p.size
    if n <= 1:
        return 0.0
    mp, mg = p.mean(), g.mean()
    vp = ((p - mp) ** 2).sum() / (n - 1)
    vg = ((g - mg) ** 2).sum() / (n - 1)
    cov = ((p - mp) * (g - mg)).sum() / (n - 1)
    a = 4 * mp * mg * cov
    b = (mp**2 + mg**2) * (vp + vg)
    if a != 0:
        return float(a / (b + EPS))
    return 1.0 if a == 0 and b == 0 else 0.0


def _s_region(pred, gt):
    h, w = gt.shape
    cy, cx = _centroid(gt)
    cy, cx = max(cy, 1), max(cx, 1)
    cy, cx = min(cy, h - 1), min(cx, w - 1)
    area = h * w
    tot = 0.0
    for ys, xs in ((slice(0, cy), slice(0, cx)), (slice(0, cy), slice(cx, w)),
                   (slice(cy, h), slice(0, cx)), (slice(cy, h), slice(cx, w))):
        g, p = gt[ys, xs], pred[ys, xs]
        tot += (g.size / area) * _ssim(p, g)
    return tot


def s_measure(pred, gt, alpha: float = 0.5):
    gt = gt.astype(np.float64)
    y = gt.mean()
    if y == 0:
        return float(1 - pred.mean())
    if y == 1:
        return float(pred.mean())
    q = alpha * _s_object(pred, gt) + (1 - alpha) * _s_region(pred, gt)
    return float(max(q, 0.0))


# --------------------------------------------------------------- weighted F
def _gauss_kernel(size=7, sigma=5.0):
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-(ax**2) / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()


def weighted_f(pred, gt, beta2: float = 1.0):
    gt = gt.astype(bool)
    if not gt.any():
        return float(pred.max() == 0)  # empty GT: only an empty prediction is correct
    dst, idx = bwdist(~gt, return_indices=True)
    err = np.abs(pred - gt.astype(np.float64))
    et = err.copy()
    et[~gt] = err[idx[0][~gt], idx[1][~gt]]
    ea = convolve(et, _gauss_kernel(7, 5.0), mode="constant")
    min_ea = np.where(gt & (ea < err), ea, err)
    b = np.where(gt, 1.0, 2.0 - np.exp(np.log(0.5) / 5 * dst))
    ew = min_ea * b
    tpw = gt.sum() - ew[gt].sum()
    fpw = ew[~gt].sum()
    r = 1 - ew[gt].mean()
    p = tpw / (EPS + tpw + fpw)
    return float((1 + beta2) * r * p / (EPS + r + beta2 * p))


# ---------------------------------------------------------------- E-measure
def e_measure(pred, gt):
    """Mean enhanced-alignment measure over 256 integer thresholds on the 8-bit prediction."""
    gt = gt.astype(bool)
    p8 = np.round(pred * 255).astype(np.uint8)
    n = gt.size
    fg_hist = np.bincount(p8[gt], minlength=256)
    bg_hist = np.bincount(p8[~gt], minlength=256)
    fg_ge = np.cumsum(fg_hist[::-1])[::-1].astype(np.float64)  # fg pixels with value >= t
    bg_ge = np.cumsum(bg_hist[::-1])[::-1].astype(np.float64)
    n_fg, n_bg = float(gt.sum()), float(n - gt.sum())
    tp, fp = fg_ge, bg_ge
    fn, tn = n_fg - tp, n_bg - fp
    if n_fg == 0:
        enh = tn / n
    elif n_bg == 0:
        enh = tp / n
    else:
        # bias-corrected alignment matrices per region type: (pred=1,gt=1),(pred=1,gt=0),(pred=0,gt=1),(pred=0,gt=0)
        mu_p = (tp + fp) / n
        mu_g = n_fg / n
        enh = np.zeros(256)
        for cnt, pv, gv in ((tp, 1, 1), (fp, 1, 0), (fn, 0, 1), (tn, 0, 0)):
            dp, dg = pv - mu_p, gv - mu_g
            align = 2 * dp * dg / (dp**2 + dg**2 + EPS)
            enh += cnt * ((align + 1) ** 2 / 4)
        enh /= n - 1 + EPS
    return float(enh.mean())


# ---------------------------------------------------------- IoU / Dice @ thr
def iou_dice(pred, gt, threshold: float = 0.5):
    b = pred >= threshold
    g = gt.astype(bool)
    inter = (b & g).sum()
    union = (b | g).sum()
    s = b.sum() + g.sum()
    return (1.0 if union == 0 else inter / union), (1.0 if s == 0 else 2 * inter / s)


class CODMetrics:
    """Accumulates per-image scores; `compute()` returns dataset means."""

    def __init__(self, threshold: float = 0.5):
        self.t = threshold
        self.rows: list[dict] = []

    def update(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        pred = np.clip(pred.astype(np.float64), 0, 1)
        gt = (gt > 0.5).astype(np.float64)
        iou, dice = iou_dice(pred, gt, self.t)
        row = dict(sm=s_measure(pred, gt), wf=weighted_f(pred, gt), em=e_measure(pred, gt),
                   mae=mae(pred, gt), miou=float(iou), dice=float(dice))
        self.rows.append(row)
        return row

    def compute(self) -> dict:
        if not self.rows:
            return {}
        return {k: float(np.mean([r[k] for r in self.rows])) for k in self.rows[0]} | {"n": len(self.rows)}
