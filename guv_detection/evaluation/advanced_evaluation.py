"""
Advanced evaluation script for object detection models using YOLO format.
Evaluates predictions vs ground truth using IoU and confidence filtering.
"""

import argparse
import os
from glob import glob
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import norm, chi2, chi2_contingency, fisher_exact

CONF_THRESH  = 0.25
IOU_THRESH   = 0.5
MIN_N_BIN    = 10   # minimum TP+FP (or TP+FN) for a bin to enter homogeneity/trend tests

MU_PER_PX = {
    "nikon_high": 0.120,   # high-magnification Nikon (quadrant tiles: _tl/_tr/_bl/_br)
    "nikon_low":  0.339,   # low-magnification  Nikon (grid tiles: _A1…_D4)
    "leica":      0.45,
}

_QUADRANT_SUFFIXES = {"_tl", "_tr", "_bl", "_br"}
_GRID_ROW_LETTERS  = set("ABCD")


def _get_mu(stem: str, is_leica: bool) -> float:
    if is_leica:
        return MU_PER_PX["leica"]
    suffix = stem[-3:]
    if suffix in _QUADRANT_SUFFIXES:
        return MU_PER_PX["nikon_high"]
    if len(suffix) == 3 and suffix[1] in _GRID_ROW_LETTERS and suffix[2].isdigit():
        return MU_PER_PX["nikon_low"]
    return MU_PER_PX["nikon_low"]   # fallback

# ── Style constants ────────────────────────────────────────────────────────────
FS_LABEL  = 14
FS_TICK   = 13
FS_LEGEND = 12
FS_ANNOT  = 8
FIGSIZE   = (12, 5)

COLOR_P   = "#084594"   # dark blue  — Precision
COLOR_R   = "#4292c6"   # mid blue   — Recall
COLOR_F1  = "#9ecae1"   # light blue — F1
COLOR_RHO = "red"       # red        — rho = P/R correction factor
COLOR_BAR_NIKON = "#1b7837"   # dark green — Nikon bars
COLOR_BAR_LEICA = "#7b2d8b"   # dark purple — Leica bars

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """Convert YOLO format (xc, yc, w, h) to (x1, y1, x2, y2) in pixels."""
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - bw / 2
    y1 = (yc * img_h) - bh / 2
    x2 = x1 + bw
    y2 = y1 + bh
    return [x1, y1, x2, y2]

def iou(box1, box2):
    """Compute Intersection over Union between two boxes."""
    x1, y1, x2, y2, conf_1 = box1
    x1g, y1g, x2g, y2g, conf_2 = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def read_gt_boxes(label_path, img_w, img_h):
    """Read ground truth boxes from YOLO txt file."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, xc, yc, w, h = map(float, parts[:5])
                    boxes.append(yolo_to_xyxy(xc, yc, w, h, img_w, img_h) + [1.0])  # Append a dummy confidence of 1.0 for GT
    return boxes

def read_pred_boxes(pred_path, img_w, img_h, conf_thresh=CONF_THRESH):
    """Read predicted boxes with confidence filtering."""
    boxes = []
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls, xc, yc, w, h, conf = map(float, parts[:6])
                    if conf >= conf_thresh:
                        boxes.append(yolo_to_xyxy(xc, yc, w, h, img_w, img_h) + [conf])
    return boxes

def get_bin(area, bin_edges, bin_labels, last_label):
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= area < bin_edges[i + 1]:
            return bin_labels[i]
    return last_label


def _guv_dim_um(box, mu: float):
    """Dataset formula: sqrt(max_dim² + min_dim²) / sqrt(2) * mu."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    max_dim, min_dim = max(w, h), min(w, h)
    return np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2) * mu


def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return phat, (center - margin) / denom, (center + margin) / denom


def benjamini_hochberg(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for a family of p-values.
    Returns (adjusted_pvals, reject) aligned to the input order.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order   = np.argsort(pvals)
    ranked  = pvals[order]
    adj_ranked = ranked * m / np.arange(1, m + 1)
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]  # enforce monotonicity
    adj_ranked = np.clip(adj_ranked, 0, 1)
    adjusted = np.empty(m)
    adjusted[order] = adj_ranked
    return adjusted, adjusted < alpha


def cramers_v(chi2_stat, n, dof_table=1):
    """
    Effect size for a K x 2 contingency table (dof_table = min(rows-1, cols-1),
    which is always 1 for a K x 2 table). Needed because a chi-square p-value
    alone conflates statistical significance with practical relevance: with a
    large N, even a trivial deviation from homogeneity becomes "significant".
    Guideline (Cohen, valid for dof_table=1): <0.10 negligible, 0.10-0.30
    small, 0.30-0.50 medium, >0.50 large.
    """
    if n <= 0:
        return float('nan')
    return float(np.sqrt(chi2_stat / (n * dof_table)))


def _cramers_v_label(v):
    if np.isnan(v):
        return 'n/a'
    if v < 0.10:
        return 'negligible'
    if v < 0.30:
        return 'small'
    if v < 0.50:
        return 'medium'
    return 'large'


def homogeneity_test(success_counts, failure_counts, bin_names, alpha=0.05, min_n=MIN_N_BIN):
    """
    Test whether a proportion (P: TP/(TP+FP), or R: TP/(TP+FN)) is homogeneous
    across size bins, using a (K x 2) contingency table of [success, failure].

    Bins with fewer than `min_n` observations (success+failure) are dropped
    first — a proportion estimated on a handful of trials carries no usable
    information about homogeneity and can distort the chi-square statistic.
    Uses a chi-square test of homogeneity by default, reporting Cramer's V as
    an effect size alongside the p-value (large N makes the chi-square test
    over-sensitive to trivial deviations, so the p-value alone is not enough).
    If more than 20% of the table's expected cell counts fall below 5
    (standard validity rule for chi-square), falls back to all-pairs Fisher
    exact tests with Benjamini-Hochberg (FDR) correction, which additionally
    identifies which specific bins differ from which.
    """
    success_counts = np.asarray(success_counts, dtype=float)
    failure_counts = np.asarray(failure_counts, dtype=float)
    n = success_counts + failure_counts
    valid = n >= min_n
    if valid.sum() < 2:
        return {'method': 'insufficient_data'}

    success_v = success_counts[valid]
    failure_v = failure_counts[valid]
    bins_v    = [b for b, ok in zip(bin_names, valid) if ok]

    table = np.array([success_v, failure_v]).T  # K x 2
    chi2_stat, p_chi2, dof, expected = chi2_contingency(table)
    low_expected_frac = float(np.mean(expected < 5))
    v = cramers_v(chi2_stat, table.sum(), dof_table=1)

    result = {
        'method': 'chi2', 'statistic': chi2_stat, 'dof': dof, 'p_value': p_chi2,
        'cramers_v': v, 'low_expected_frac': low_expected_frac, 'pairwise': None, 'alpha': alpha,
    }

    if low_expected_frac > 0.2:
        pairs, pvals = [], []
        for (i, bi), (j, bj) in combinations(enumerate(bins_v), 2):
            table_2x2 = [[success_v[i], failure_v[i]], [success_v[j], failure_v[j]]]
            _, p_fisher = fisher_exact(table_2x2)
            pairs.append((bi, bj))
            pvals.append(p_fisher)
        adjusted, reject = benjamini_hochberg(pvals, alpha=alpha)
        result['method']   = 'fisher_pairwise_fdr'
        result['pairwise'] = list(zip(pairs, pvals, adjusted, reject))

    return result


def cochran_armitage_trend(successes, totals, scores=None, min_n=MIN_N_BIN):
    """
    Cochran-Armitage trend test: tests whether a proportion (e.g. P or R)
    changes monotonically with an ordered covariate (e.g. bin diameter),
    which is a more specific and more powerful test than an unordered
    chi-square/Fisher homogeneity test when the bins have a natural order.
    Bins with fewer than `min_n` trials are dropped. Returns (z, p_value);
    z > 0 means the proportion increases with the score (i.e. with diameter).
    """
    successes = np.asarray(successes, dtype=float)
    totals    = np.asarray(totals, dtype=float)
    valid = totals >= min_n
    if valid.sum() < 2:
        return float('nan'), float('nan')
    successes = successes[valid]
    totals    = totals[valid]
    scores = np.arange(len(successes), dtype=float) if scores is None else np.asarray(scores, dtype=float)[valid]

    N     = totals.sum()
    p_bar = successes.sum() / N
    t_bar = np.average(scores, weights=totals)
    num   = np.sum(totals * (scores - t_bar) * (successes / totals - p_bar))
    denom = p_bar * (1 - p_bar) * np.sum(totals * (scores - t_bar) ** 2)
    if denom <= 0:
        return float('nan'), float('nan')
    z = num / np.sqrt(denom)
    # z**2 ~ chi2(1) under H0; this is already the two-sided p-value (no extra *2).
    p_value = 1 - chi2.cdf(z**2, df=1)
    return z, p_value


def _bin_score(label, width):
    """Numeric midpoint of a bin label like '10-20', or lo+width/2 for an open-ended '>100'."""
    if label.startswith('>'):
        return float(label[1:]) + width / 2
    lo, hi = label.split('-')
    return (float(lo) + float(hi)) / 2


def _print_homogeneity(name, result):
    if result['method'] == 'insufficient_data':
        print(f"  Homogeneity test ({name}): insufficient data (fewer than 2 bins with n>={MIN_N_BIN}) — skipped")
        return
    v = result['cramers_v']
    v_label = _cramers_v_label(v)
    if result['method'] == 'chi2':
        verdict = 'NOT homogeneous (p<0.05) -> systematic variation across bins' \
                  if result['p_value'] < 0.05 else 'homogeneous (p>=0.05) -> consistent with sampling noise'
        print(f"  Homogeneity test ({name}): chi2={result['statistic']:.3f}, dof={result['dof']}, "
              f"p={result['p_value']:.4f}, Cramer's V={v:.3f} ({v_label}) -> {verdict}")
    else:
        print(f"  Homogeneity test ({name}): chi2 assumptions violated "
              f"({result['low_expected_frac']*100:.0f}% of expected cell counts < 5), "
              f"omnibus Cramer's V={v:.3f} ({v_label}) -> "
              f"pairwise Fisher exact tests, Benjamini-Hochberg FDR corrected (alpha={result['alpha']}):")
        for (b1, b2), p_raw, p_adj, sig in result['pairwise']:
            flag = " *** DIFFERENT ***" if sig else ""
            print(f"    {b1} vs {b2}: p_raw={p_raw:.4f}, p_adj={p_adj:.4f}{flag}")


def _print_trend(name, z, p_value):
    if np.isnan(z):
        print(f"  Trend test ({name}): insufficient data — skipped")
        return
    direction = 'increasing' if z > 0 else 'decreasing'
    verdict = f"significant {direction} trend with diameter (p<0.05)" if p_value < 0.05 \
              else "no significant trend with diameter (p>=0.05)"
    print(f"  Cochran-Armitage trend test ({name}): z={z:.3f}, p={p_value:.4f} -> {verdict}")


def _fit_log_rho_trend(tp, fp, fn, scores):
    """
    OLS slope of log(rho) = log(P) - log(R) vs. bin score, over bins with
    TP > 0. Only meant to be called once per bootstrap resample (see
    bootstrap_rho): fitting on raw resampled counts, rather than on an
    analytic combination of Var(logP) and Var(logR), sidesteps having to
    assume logP and logR are independent — they are not, since both share
    the same TP count in their numerator.
    """
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    scores = np.asarray(scores, dtype=float)

    valid = tp > 0
    if valid.sum() < 2:
        return float('nan')
    P = tp[valid] / (tp[valid] + fp[valid])
    R = tp[valid] / (tp[valid] + fn[valid])
    log_rho = np.log(P / R)
    slope, _ = np.polyfit(scores[valid], log_rho, 1)
    return float(slope)


def bootstrap_rho(per_image_stats, bin_names, trend_bins, trend_scores, n_boot=2000, seed=42):
    """
    Cluster bootstrap by image: resamples images with replacement (preserving
    within-image correlation between boxes) and, on each resample, recomputes
    rho globally, rho per bin, and the OLS trend slope of log(rho) vs. the
    bin score restricted to `trend_bins`/`trend_scores` (a fixed, pre-filtered
    subset — see MIN_N_BIN — so every replicate fits the trend on the same
    physical bins). Returns percentile (2.5/97.5) summaries.
    """
    rng = np.random.default_rng(seed)
    images = list(per_image_stats.keys())
    n_img = len(images)

    boot_rho_global = np.full(n_boot, np.nan)
    boot_rho_bins   = {b: np.full(n_boot, np.nan) for b in bin_names}
    boot_slope      = np.full(n_boot, np.nan)

    for i in range(n_boot):
        sample = rng.choice(images, size=n_img, replace=True)

        tp_g = fp_g = fn_g = 0.0
        tp_bin = {b: 0.0 for b in bin_names}
        fp_bin = {b: 0.0 for b in bin_names}
        fn_bin = {b: 0.0 for b in bin_names}

        for img in sample:
            for b, counts in per_image_stats[img].items():
                if b not in tp_bin:
                    continue
                tp_bin[b] += counts['TP']; fp_bin[b] += counts['FP']; fn_bin[b] += counts['FN']
                tp_g += counts['TP'];      fp_g += counts['FP'];      fn_g += counts['FN']

        if (tp_g + fp_g) > 0 and (tp_g + fn_g) > 0:
            p_g = tp_g / (tp_g + fp_g)
            r_g = tp_g / (tp_g + fn_g)
            if r_g > 0:
                boot_rho_global[i] = p_g / r_g

        for b in bin_names:
            n_p, n_r = tp_bin[b] + fp_bin[b], tp_bin[b] + fn_bin[b]
            if n_p > 0 and n_r > 0:
                p_b = tp_bin[b] / n_p
                r_b = tp_bin[b] / n_r
                if r_b > 0:
                    boot_rho_bins[b][i] = p_b / r_b

        tp_t = np.array([tp_bin[b] for b in trend_bins])
        fp_t = np.array([fp_bin[b] for b in trend_bins])
        fn_t = np.array([fn_bin[b] for b in trend_bins])
        boot_slope[i] = _fit_log_rho_trend(tp_t, fp_t, fn_t, trend_scores)

    def summarize(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {'mean': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'), 'n_valid': 0}
        return {'mean': float(np.mean(arr)), 'ci_lo': float(np.percentile(arr, 2.5)),
                'ci_hi': float(np.percentile(arr, 97.5)), 'n_valid': int(arr.size)}

    return {
        'n_boot': n_boot,
        'global': summarize(boot_rho_global),
        'per_bin': {b: summarize(boot_rho_bins[b]) for b in bin_names},
        'trend_slope': summarize(boot_slope),
    }


def _print_bootstrap_rho(result):
    n_boot = result['n_boot']
    g = result['global']
    print(f"  Bootstrap rho (global, cluster-by-image, n_boot={n_boot}): "
          f"mean={g['mean']:.3f}, 95% CI=[{g['ci_lo']:.3f}, {g['ci_hi']:.3f}] "
          f"(valid resamples: {g['n_valid']}/{n_boot})")
    for b, r in result['per_bin'].items():
        if r['n_valid'] > 0:
            print(f"    {b}: mean={r['mean']:.3f}, 95% CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
                  f"(valid resamples: {r['n_valid']}/{n_boot})")
        else:
            print(f"    {b}: no valid resamples")
    s = result['trend_slope']
    if s['n_valid'] > 0:
        excludes_zero = s['ci_lo'] > 0 or s['ci_hi'] < 0
        verdict = 'excludes 0 -> significant trend' if excludes_zero else 'includes 0 -> no significant trend'
        print(f"  Bootstrap trend slope of log(rho) vs. diameter (bins with n>={MIN_N_BIN}): "
              f"mean={s['mean']:.4f}, 95% CI=[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] -> {verdict}")
    else:
        print("  Bootstrap trend slope of log(rho): insufficient data — skipped")


def _draw_eval_plot(bin_names, precisions, recalls, f1_scores, rho,
                    num_detected, num_gt, xlabel, title, fig_num,
                    legend_outside=False, color_bar=COLOR_BAR_NIKON):
    """Shared plotting routine for px and µm evaluation figures."""
    x        = np.arange(len(bin_names))
    width    = 0.35
    y_offset = 5

    fig, ax1 = plt.subplots(figsize=FIGSIZE, num=fig_num)

    # ── Metric lines ──────────────────────────────────────────────────────
    ax1.plot(x, precisions, marker='o', label='Precision', color=COLOR_P,  linewidth=1.8)
    ax1.plot(x, recalls,    marker='s', label='Recall',    color=COLOR_R,  linewidth=1.8)
    ax1.plot(x, f1_scores,  marker='^', label='F1 Score',  color=COLOR_F1, linewidth=1.8)
    ax1.plot(x, rho,        marker='d', label='ρ (P/R)',   color=COLOR_RHO, linewidth=1.8, linestyle='--')

    rho_arr     = np.asarray(rho, dtype=float)
    finite_rho  = rho_arr[np.isfinite(rho_arr)]
    y_top       = max(1.15, float(np.max(finite_rho)) * 1.1) if finite_rho.size > 0 else 1.15
    ax1.set_ylim(0, y_top)
    ax1.set_ylabel('Precision / Recall / F1 / ρ', fontsize=FS_LABEL)
    ax1.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_names, rotation=45, fontsize=FS_TICK)
    ax1.tick_params(axis='y', labelsize=FS_TICK)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.grid(True, linestyle=':', linewidth=0.8, color='#bbbbbb', zorder=0)
    ax1.set_axisbelow(True)

    # ── Count bars (secondary axis) ───────────────────────────────────────
    ax2 = ax1.twinx()
    bars_det = ax2.bar(x - width / 2, num_detected, width,
                       color=color_bar, alpha=0.9, edgecolor='white', linewidth=0.5,
                       label='Detected GUV')
    bars_gt  = ax2.bar(x + width / 2, num_gt, width,
                       color=color_bar, alpha=0.45, edgecolor='white', linewidth=0.5,
                       hatch='//', label='GT GUV')

    ax2.set_ylim(0, 1200)
    ax2.set_ylabel('Number of GUV', fontsize=FS_LABEL, color=color_bar)
    ax2.tick_params(axis='y', labelsize=FS_TICK, colors=color_bar)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_edgecolor(color_bar)
    ax2.yaxis.grid(True, linestyle=':', linewidth=0.8, color=color_bar, alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)

    # metric lines always on top of bars
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    if legend_outside:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=FS_LEGEND,
                   loc='upper left', bbox_to_anchor=(1.08, 1), borderaxespad=0)
    else:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
                   fontsize=FS_LEGEND, loc='upper left')

    # plt.title(title, fontsize=FS_LABEL + 2)
    plt.tight_layout()
    plt.show()

def evaluate(folder, model_size, modality):
    image_dir = os.path.join(folder, 'images')
    label_dir = os.path.join(folder, 'labels')
    pred_dir  = os.path.join(folder, f'predict_{model_size}')

    assert os.path.isdir(image_dir), f"Missing images folder: {image_dir}"
    assert os.path.isdir(label_dir), f"Missing labels folder: {label_dir}"
    assert os.path.isdir(pred_dir),  f"Missing predictions folder: {pred_dir}"

    is_leica  = "leica" in folder.lower()
    color_bar = COLOR_BAR_LEICA if is_leica else COLOR_BAR_NIKON

    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png")))
    image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    # pixel bins
    px_edges  = list(range(0, 110, 10))
    px_labels = [f"{px_edges[i]}-{px_edges[i+1]}" for i in range(len(px_edges) - 1)]
    px_last   = ">100"

    # µm bins
    um_edges  = list(range(0, 40, 5))
    um_labels = [f"{um_edges[i]}-{um_edges[i+1]}" for i in range(len(um_edges) - 1)]
    um_last   = ">35"

    stats = {'TP': 0, 'FP': 0, 'FN': 0}
    stats_per_bin    = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'GT_total': 0})
    stats_per_bin_um = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'GT_total': 0})
    per_image_stats_um = {}   # img_name -> {um_bin: {'TP','FP','FN'}}, for the cluster bootstrap
    per_image_stats_px = {}   # img_name -> {px_bin: {'TP','FP','FN'}}, for the cluster bootstrap
    all_gt_count = 0

    for img_name in image_names:
        img_path = os.path.join(image_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_name}.png")
        img = Image.open(img_path)
        img_w, img_h = img.size

        mu = _get_mu(img_name, is_leica)
        per_image_stats_um[img_name] = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
        per_image_stats_px[img_name] = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

        gt_path = os.path.join(label_dir, f"{img_name}.txt")
        pred_path = os.path.join(pred_dir, 'labels', f"{img_name}.txt")

        gt_boxes = read_gt_boxes(gt_path, img_w, img_h)
        pred_boxes = read_pred_boxes(pred_path, img_w, img_h)

        # GT bins (px and µm)
        for gb in gt_boxes:
            gb_w = gb[2] - gb[0]
            gb_h = gb[3] - gb[1]
            dim_px = (gb_w + gb_h) / 2
            dim_um = _guv_dim_um(gb, mu)
            g_bin    = get_bin(dim_px, px_edges, px_labels, px_last)
            g_bin_um = get_bin(dim_um, um_edges, um_labels, um_last)
            stats_per_bin[g_bin]['GT_total']       += 1
            stats_per_bin_um[g_bin_um]['GT_total'] += 1
            all_gt_count += 1

        matched_gt = set()
        for pb in pred_boxes:
            p_w = pb[2] - pb[0]
            p_h = pb[3] - pb[1]
            dim_px = (p_w + p_h) / 2
            dim_um = _guv_dim_um(pb, mu)
            p_bin    = get_bin(dim_px, px_edges, px_labels, px_last)
            p_bin_um = get_bin(dim_um, um_edges, um_labels, um_last)

            best_iou = 0
            best_gt  = -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou_val = iou(pb, gb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt  = i

            conf = pb[4] if len(pb) > 4 else 1.0
            if best_iou >= IOU_THRESH:
                matched_gt.add(best_gt)
                stats['TP'] += 1
                stats_per_bin[p_bin]['TP']          += 1
                stats_per_bin_um[p_bin_um]['TP']    += 1
                per_image_stats_um[img_name][p_bin_um]['TP'] += 1
                per_image_stats_px[img_name][p_bin]['TP']    += 1
            else:
                stats['FP'] += 1
                stats_per_bin[p_bin]['FP']          += 1
                stats_per_bin_um[p_bin_um]['FP']    += 1
                per_image_stats_um[img_name][p_bin_um]['FP'] += 1
                per_image_stats_px[img_name][p_bin]['FP']    += 1

        # FN
        for i, gb in enumerate(gt_boxes):
            if i not in matched_gt:
                gb_w = gb[2] - gb[0]
                gb_h = gb[3] - gb[1]
                dim_px = (gb_w + gb_h) / 2
                dim_um = _guv_dim_um(gb, mu)
                g_bin    = get_bin(dim_px, px_edges, px_labels, px_last)
                g_bin_um = get_bin(dim_um, um_edges, um_labels, um_last)
                stats['FN'] += 1
                stats_per_bin[g_bin]['FN']          += 1
                stats_per_bin_um[g_bin_um]['FN']    += 1
                per_image_stats_um[img_name][g_bin_um]['FN'] += 1
                per_image_stats_px[img_name][g_bin]['FN']    += 1


    def bin_sort_key(b):
        try:
            return int(b.split('-')[0])
        except Exception:
            return 9999

    def _compute_metrics(spb):
        all_bins = sorted(spb.keys(), key=bin_sort_key)
        prec, rec, f1s, rho, n_det, n_gt = [], [], [], [], [], []
        for b in all_bins:
            tp = spb[b]['TP']; fp = spb[b]['FP']; fn = spb[b]['FN']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            rho_val = p / r if r > 0 else np.nan
            prec.append(p); rec.append(r); f1s.append(f); rho.append(rho_val)
            n_det.append(tp + fp); n_gt.append(spb[b]['GT_total'])
        return all_bins, prec, rec, f1s, rho, np.array(n_det), np.array(n_gt)

    def compute_ap(recall, precision):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x    = np.linspace(0, 1, 101)
        func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        return func(np.interp(x, mrec, mpre), x)

    # ── Global rho (P/R correction factor) ──────────────────────────────────
    p_global   = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
    r_global   = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0
    rho_global = p_global / r_global if r_global > 0 else float('nan')
    print(f"Global P={p_global:.3f}, R={r_global:.3f}, rho (P/R)={rho_global:.3f}\n")

    # ── Pixel plot ────────────────────────────────────────────────────────
    bin_names, precisions, recalls, f1_scores, rhos, num_detected, num_gt = \
        _compute_metrics(stats_per_bin)
    tp_arr = np.array([stats_per_bin[b]['TP'] for b in bin_names])
    fp_arr = np.array([stats_per_bin[b]['FP'] for b in bin_names])
    fn_arr = np.array([stats_per_bin[b]['FN'] for b in bin_names])

    print(f"Total GT: {sum(num_gt)}, Total Detected: {sum(num_detected)}\n")
    print("Bin-wise Evaluation (pixels):")
    for j in range(len(bin_names)):
        _, p_lo, p_hi = wilson_ci(tp_arr[j], tp_arr[j] + fp_arr[j])
        _, r_lo, r_hi = wilson_ci(tp_arr[j], tp_arr[j] + fn_arr[j])
        print(f"  Bin {bin_names[j]}: GT={num_gt[j]}, Det={num_detected[j]}, "
              f"P={precisions[j]:.3f} [{p_lo:.3f}-{p_hi:.3f}], "
              f"R={recalls[j]:.3f} [{r_lo:.3f}-{r_hi:.3f}], "
              f"F1={f1_scores[j]:.3f}, rho={rhos[j]:.3f}")
    print(f"Overall AP (px): {compute_ap(recalls, precisions):.4f}")
    _print_homogeneity("Precision, px bins: TP vs FP", homogeneity_test(tp_arr, fp_arr, bin_names))
    _print_homogeneity("Recall, px bins: TP vs FN", homogeneity_test(tp_arr, fn_arr, bin_names))
    px_scores = np.array([_bin_score(b, 10) for b in bin_names])
    _print_trend("Precision, px bins", *cochran_armitage_trend(tp_arr, tp_arr + fp_arr, scores=px_scores))
    _print_trend("Recall, px bins",    *cochran_armitage_trend(tp_arr, tp_arr + fn_arr, scores=px_scores))
    print()

    # ── Bootstrap on rho (px bins) — secondary/diagnostic axis; note that   ──
    # ── mu_per_px differs across acquisition sources, so a "px bin" mixes   ──
    # ── physically different diameters if the test set spans several scopes ──
    trend_mask_px   = (tp_arr + fp_arr >= MIN_N_BIN) & (tp_arr + fn_arr >= MIN_N_BIN)
    trend_bins_px   = [b for b, ok in zip(bin_names, trend_mask_px) if ok]
    trend_scores_px = px_scores[trend_mask_px]
    boot_result_px  = bootstrap_rho(per_image_stats_px, bin_names, trend_bins_px, trend_scores_px)
    print("Bootstrap evaluation, cluster-by-image (px bins):")
    _print_bootstrap_rho(boot_result_px)
    print()

    _draw_eval_plot(
        bin_names, precisions, recalls, f1_scores, rhos, num_detected, num_gt,
        xlabel='GUV Size Range (pixels)',
        title=f'Precision / Recall / F1 by BBox Size (px) — {modality} YOLOv11_{model_size}',
        fig_num=f"Eval-px-{modality}-{model_size}",
        color_bar=color_bar,
    )

    # ── µm plot ───────────────────────────────────────────────────────────
    bin_names_um, prec_um, rec_um, f1_um, rho_um, ndet_um, ngt_um = \
        _compute_metrics(stats_per_bin_um)
    tp_um = np.array([stats_per_bin_um[b]['TP'] for b in bin_names_um])
    fp_um = np.array([stats_per_bin_um[b]['FP'] for b in bin_names_um])
    fn_um = np.array([stats_per_bin_um[b]['FN'] for b in bin_names_um])

    print("Bin-wise Evaluation (µm):")
    for j in range(len(bin_names_um)):
        _, p_lo, p_hi = wilson_ci(tp_um[j], tp_um[j] + fp_um[j])
        _, r_lo, r_hi = wilson_ci(tp_um[j], tp_um[j] + fn_um[j])
        print(f"  Bin {bin_names_um[j]} µm: GT={ngt_um[j]}, Det={ndet_um[j]}, "
              f"P={prec_um[j]:.3f} [{p_lo:.3f}-{p_hi:.3f}], "
              f"R={rec_um[j]:.3f} [{r_lo:.3f}-{r_hi:.3f}], "
              f"F1={f1_um[j]:.3f}, rho={rho_um[j]:.3f}")
    print(f"Overall AP (µm): {compute_ap(rec_um, prec_um):.4f}")
    _print_homogeneity("Precision, µm bins: TP vs FP", homogeneity_test(tp_um, fp_um, bin_names_um))
    _print_homogeneity("Recall, µm bins: TP vs FN", homogeneity_test(tp_um, fn_um, bin_names_um))
    um_scores = np.array([_bin_score(b, 5) for b in bin_names_um])
    _print_trend("Precision, µm bins", *cochran_armitage_trend(tp_um, tp_um + fp_um, scores=um_scores))
    _print_trend("Recall, µm bins",    *cochran_armitage_trend(tp_um, tp_um + fn_um, scores=um_scores))
    print()

    # ── Bootstrap on rho (µm bins) — the physically meaningful axis, since   ──
    # ── mu_per_px differs across acquisition sources (Leica/Nikon high/low)  ──
    trend_mask_um   = (tp_um + fp_um >= MIN_N_BIN) & (tp_um + fn_um >= MIN_N_BIN)
    trend_bins_um   = [b for b, ok in zip(bin_names_um, trend_mask_um) if ok]
    trend_scores_um = um_scores[trend_mask_um]
    boot_result_um  = bootstrap_rho(per_image_stats_um, bin_names_um, trend_bins_um, trend_scores_um)
    print("Bootstrap evaluation, cluster-by-image (µm bins):")
    _print_bootstrap_rho(boot_result_um)
    print()

    _draw_eval_plot(
        bin_names_um, prec_um, rec_um, f1_um, rho_um, ndet_um, ngt_um,
        xlabel='GUV Size Range (µm)',
        title=f'Precision / Recall / F1 by BBox Size (µm) — {modality} YOLOv11_{model_size}',
        fig_num=f"Eval-um-{modality}-{model_size}",
        legend_outside=True,
        color_bar=color_bar,
    )
            
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model on a folder of images")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/",
                        help="Path to the root folder (parent of DATA_training_*_txt)")
    parser.add_argument("--subfolder", type=str, default="test",
                        help="Subfolder inside DATA_training_{modality}_txt, e.g. 'test' or 'Leica-tot/test'")
    args = parser.parse_args()

    folder = os.path.join(args.folder, f'DATA_training_{args.modality}_txt', args.subfolder)
    evaluate(folder, args.model_size, args.modality)