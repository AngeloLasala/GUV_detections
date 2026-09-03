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

CONF_THRESH  = 0.25
IOU_THRESH   = 0.5
MIN_N_BIN    = 10   # minimum TP+FP (or TP+FN) for a bin to enter the bootstrap trend fit

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
COLOR_RHO = "red"       # red        — rho = P/R correction factor, all bins
COLOR_RHO_EXCL = "darkorange"  # orange — rho = P/R correction factor, excl. smallest bin
COLOR_OBS = "brown"     # brown      — observed per-bin scatter points
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


def _bin_score(label, width):
    """Numeric midpoint of a bin label like '10-20', or lo+width/2 for an open-ended '>100'."""
    if label.startswith('>'):
        return float(label[1:]) + width / 2
    lo, hi = label.split('-')
    return (float(lo) + float(hi)) / 2


def _fit_log_rho_trend(tp, fp, fn, scores):
    """
    Linear regression slope+intercept of log(rho) = log(P) - log(R) vs. bin score, over
    bins with TP > 0. Only meant to be called once per bootstrap resample
    (see bootstrap_rho): fitting on raw resampled counts, rather than on an
    analytic combination of Var(logP) and Var(logR), sidesteps having to
    assume logP and logR are independent — they are not, since both share
    the same TP count in their numerator. Returns (slope, intercept).
    """
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    scores = np.asarray(scores, dtype=float)

    valid = tp > 0
    if valid.sum() < 2:
        return float('nan'), float('nan')
    P = tp[valid] / (tp[valid] + fp[valid])
    R = tp[valid] / (tp[valid] + fn[valid])
    log_rho = np.log(P / R)
    slope, intercept = np.polyfit(scores[valid], log_rho, 1)
    return float(slope), float(intercept)


def _observed_log_rho(tp, fp, fn, scores, bin_names):
    """Per-bin observed log(rho) = log(P) - log(R), restricted to bins with TP > 0."""
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    scores = np.asarray(scores, dtype=float)
    valid = tp > 0
    P = tp[valid] / (tp[valid] + fp[valid])
    R = tp[valid] / (tp[valid] + fn[valid])
    log_rho = np.log(P / R)
    weights = (tp + fp + fn)[valid]
    names = [b for b, ok in zip(bin_names, valid) if ok]
    return scores[valid], log_rho, weights, names


def bootstrap_rho(per_image_stats, bin_names, trend_variants, n_boot=2000, seed=42):
    """
    Cluster bootstrap by image: resamples images with replacement (preserving
    within-image correlation between boxes) and, on each resample, recomputes
    rho globally, rho per bin, and — for every entry in `trend_variants`
    (label -> (trend_bins, trend_scores)) — the linear regression trend line (slope +
    intercept) of log(rho) vs. the bin score restricted to that bin subset.
    Fitting several variants (e.g. "all bins" and "excl. smallest bin") inside
    the same resampling pass means they share the same 2000 resamples instead
    of each needing its own bootstrap run. Each variant's trend line is also
    evaluated on a grid spanning its own trend_scores, so a 95% pointwise CI
    band can be plotted around the fit (see `_draw_rho_trend_plot`). Returns
    percentile (2.5/97.5) summaries.
    """
    rng = np.random.default_rng(seed)
    images = list(per_image_stats.keys())
    n_img = len(images)

    boot_rho_global = np.full(n_boot, np.nan)
    boot_rho_bins   = {b: np.full(n_boot, np.nan) for b in bin_names}
    boot_slope      = {label: np.full(n_boot, np.nan) for label in trend_variants}
    boot_intercept  = {label: np.full(n_boot, np.nan) for label in trend_variants}

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

        for label, (tb, ts) in trend_variants.items():
            tp_t = np.array([tp_bin[b] for b in tb])
            fp_t = np.array([fp_bin[b] for b in tb])
            fn_t = np.array([fn_bin[b] for b in tb])
            boot_slope[label][i], boot_intercept[label][i] = _fit_log_rho_trend(tp_t, fp_t, fn_t, ts)

    def summarize(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {'mean': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'), 'n_valid': 0}
        return {'mean': float(np.mean(arr)), 'ci_lo': float(np.percentile(arr, 2.5)),
                'ci_hi': float(np.percentile(arr, 97.5)), 'n_valid': int(arr.size)}

    trends = {}
    for label, (tb, ts) in trend_variants.items():
        ts_arr = np.asarray(ts, dtype=float)
        slope_arr, intercept_arr = boot_slope[label], boot_intercept[label]
        valid_fit = np.isfinite(slope_arr) & np.isfinite(intercept_arr)
        # Pointwise 95% CI band for the fitted line, evaluated on a grid
        # spanning ts. Needs >=2 distinct scores and >=1 valid bootstrap fit.
        if ts_arr.size >= 2 and valid_fit.sum() > 0:
            x_grid  = np.linspace(ts_arr.min(), ts_arr.max(), 100)
            y_preds = slope_arr[valid_fit, None] * x_grid[None, :] + intercept_arr[valid_fit, None]
            trend_line = {
                'x_grid': x_grid.tolist(),
                'y_mean': np.mean(y_preds, axis=0).tolist(),
                'y_lo':   np.percentile(y_preds, 2.5, axis=0).tolist(),
                'y_hi':   np.percentile(y_preds, 97.5, axis=0).tolist(),
            }
        else:
            trend_line = None
        trends[label] = {
            'bins': tb, 'scores': ts,
            'slope': summarize(slope_arr), 'intercept': summarize(intercept_arr),
            'trend_line': trend_line,
            # Raw per-resample fit, kept (not just the 95% summary) so a
            # specific point can be checked at any CI level without
            # re-running the bootstrap — see _print_ci_outliers.
            'boot_slope': slope_arr[valid_fit], 'boot_intercept': intercept_arr[valid_fit],
        }

    return {
        'n_boot': n_boot,
        'global': summarize(boot_rho_global),
        'per_bin': {b: summarize(boot_rho_bins[b]) for b in bin_names},
        'trends': trends,
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
    for label, t in result['trends'].items():
        s = t['slope']
        if s['n_valid'] > 0:
            excludes_zero = s['ci_lo'] > 0 or s['ci_hi'] < 0
            verdict = 'excludes 0 -> significant trend' if excludes_zero else 'includes 0 -> no significant trend'
            print(f"  Bootstrap trend slope of log(rho) vs. diameter [{label}] (bins with n>={MIN_N_BIN}): "
                  f"mean={s['mean']:.4f}, 95% CI=[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] -> {verdict}")
        else:
            print(f"  Bootstrap trend slope of log(rho) [{label}]: insufficient data — skipped")


def _print_ci_outliers(obs_scores, obs_log_rho, obs_names, variants, unit_label,
                       extra_levels=(96, 97, 98, 99, 99.5, 99.9)):
    """
    Console-only diagnostic (no plot): for each fitted variant, checks every
    observed per-bin log(rho) point against that variant's own bootstrap
    95% CI band, evaluated exactly at the bin's score (direct evaluation of
    the raw per-resample fits `boot_slope`/`boot_intercept` at that x — not
    an interpolation of the 100-point plotting grid). A point outside the
    band is not necessarily anomalous: the band is a CI on where the fitted
    *line* lies (parameter uncertainty), not a prediction interval for
    individual data points, so it's expected to be narrower than the
    natural bin-to-bin scatter. For every outside point, widens the CI
    level step by step (`extra_levels`) and reports the first level at
    which the point falls inside — i.e. "how far outside the 95% band" in
    CI-level terms.
    """
    print(f"CI-outlier check ({unit_label} bins) — reminder: the band is a 95% CI on the "
          f"fitted line (parameter uncertainty), not a prediction interval for individual "
          f"bins, so points sitting outside it are not automatically a red flag:")

    any_outlier = False
    for v in variants:
        trend_result = v['trend_result']
        boot_slope, boot_intercept = trend_result['boot_slope'], trend_result['boot_intercept']
        if boot_slope.size == 0:
            continue
        domain_bins = set(trend_result['bins'])

        for x_b, y_b, name in zip(obs_scores, obs_log_rho, obs_names):
            if name not in domain_bins:
                continue
            y_boot = boot_slope * x_b + boot_intercept
            lo95, hi95 = np.percentile(y_boot, [2.5, 97.5])
            if lo95 <= y_b <= hi95:
                continue

            any_outlier = True
            found_level = None
            for level in extra_levels:
                tail = (100 - level) / 2
                lo, hi = np.percentile(y_boot, [tail, 100 - tail])
                if lo <= y_b <= hi:
                    found_level = level
                    break
            if found_level is not None:
                print(f"    [{v['label']}] bin {name}: log(ρ)={y_b:.4f} outside 95% CI "
                      f"[{lo95:.4f}, {hi95:.4f}] -> falls inside at CI={found_level:g}%")
            else:
                print(f"    [{v['label']}] bin {name}: log(ρ)={y_b:.4f} outside 95% CI "
                      f"[{lo95:.4f}, {hi95:.4f}] -> still outside even at {extra_levels[-1]:g}% CI")

    if not any_outlier:
        print("    none — every observed bin falls inside its variant's 95% CI band")
    print()


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


def _draw_rho_trend_plot(obs_scores, obs_log_rho, obs_weights, obs_names,
                         variants, xlabel, title, fig_num):
    """
    log(rho) = log(P) - log(R) vs. bin diameter, with one or more trend-fit
    variants overlaid on the same axes (e.g. "all bins" vs. "excl. smallest
    bin"), each in its own color, sharing one observed-points scatter
    (marker area ~ TP+FP+FN in that bin) and one legend/annotation box.

    `variants` is a list of dicts, each with:
        'label'        : str, shown in the legend and annotation box
        'color'        : matplotlib color for that variant's fit line + CI band
        'slope'        : float, observed-data regression slope
        'intercept'    : float, observed-data regression intercept
        'rho'          : float, pooled rho (P/R) over the bins this variant fits on
        'trend_result' : one entry of `bootstrap_rho(...)['trends']`
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, num=fig_num)

    summary_lines = []
    any_plotted = False
    for v in variants:
        trend_line = v['trend_result'].get('trend_line')
        if trend_line is None:
            print(f"  [{title}] insufficient data for variant '{v['label']}' — skipped")
            continue
        any_plotted = True

        x_grid = np.asarray(trend_line['x_grid'])
        y_lo   = np.asarray(trend_line['y_lo'])
        y_hi   = np.asarray(trend_line['y_hi'])

        ax.fill_between(x_grid, y_lo, y_hi, color=v['color'], alpha=0.08, zorder=1)
        ax.plot(x_grid, v['slope'] * x_grid + v['intercept'], color=v['color'], linewidth=3,
                 zorder=3, label=v['label'])

        s = v['trend_result']['slope']
        excludes_zero = np.isfinite(s['ci_lo']) and (s['ci_lo'] > 0 or s['ci_hi'] < 0)
        verdict = 'significant' if excludes_zero else 'not significant'
        summary_lines.append(
            f"{v['label']}: slope={s['mean']:.4f}  95% CI [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] -> {verdict}, "
            f"ρ={v['rho']:.3f}")

    if not any_plotted:
        print(f"  [{title}] insufficient data for a trend-line plot — skipped")
        plt.close(fig)
        return

    obs_weights = np.asarray(obs_weights, dtype=float)
    sizes = 40 + 260 * (obs_weights / obs_weights.max()) if obs_weights.max() > 0 else 60
    ax.scatter(obs_scores, obs_log_rho, s=sizes, color=COLOR_OBS, edgecolor='white',
               linewidth=0.8, zorder=5, label='Observed bins (size ~ N)')
    for xi, yi, name in zip(obs_scores, obs_log_rho, obs_names):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=FS_ANNOT)

    ax.text(0.02, 0.02, "\n".join(summary_lines),
            transform=ax.transAxes, fontsize=FS_ANNOT + 2, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel('log(ρ) = log(P) − log(R)', fontsize=FS_LABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.8, color='#bbbbbb', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEGEND, loc='best')
    # plt.title(title, fontsize=FS_LABEL + 2)
    plt.tight_layout()
    plt.show()


def _run_rho_trend_analysis(per_image_stats, bin_names, tp_arr, fp_arr, fn_arr, scores,
                            unit_label, xlabel, modality, model_size):
    """
    Bootstrap + plot the log(rho) vs. diameter trend, in two variants by
    default: using all bins that pass MIN_N_BIN, and excluding the smallest
    (leftmost) qualifying bin — the smallest liposomes are the ones most
    exposed to detection/tiling artifacts, so it's worth checking whether
    that single bin is driving the trend. Both variants are fit inside the
    same bootstrap resampling pass (see bootstrap_rho) and drawn together on
    a single plot (different color per fit), so they're directly comparable.
    """
    trend_mask   = (tp_arr + fp_arr >= MIN_N_BIN) & (tp_arr + fn_arr >= MIN_N_BIN)
    trend_bins   = [b for b, ok in zip(bin_names, trend_mask) if ok]
    trend_scores = scores[trend_mask]
    tp_t, fp_t, fn_t = tp_arr[trend_mask], fp_arr[trend_mask], fn_arr[trend_mask]

    trend_variants = {'all bins': (trend_bins, trend_scores)}
    if len(trend_bins) >= 3:
        trend_variants['excl. smallest bin'] = (trend_bins[1:], trend_scores[1:])

    boot_result = bootstrap_rho(per_image_stats, bin_names, trend_variants)
    print(f"Bootstrap evaluation, cluster-by-image ({unit_label} bins):")
    _print_bootstrap_rho(boot_result)
    print()

    # Observed points always come from the full ("all bins") set — the
    # "excl. smallest bin" fit is overlaid on the very same scatter, just
    # computed without the leftmost point, so both fits stay comparable
    # against one shared set of dots.
    obs_scores, obs_logrho, obs_w, obs_names = _observed_log_rho(tp_t, fp_t, fn_t, trend_scores, trend_bins)

    variant_colors = {'all bins': COLOR_RHO, 'excl. smallest bin': COLOR_RHO_EXCL}
    variant_display_labels = {'all bins': 'all bins', 'excl. smallest bin': 'w/o smallest'}
    variants = []
    for label, (tb, ts) in trend_variants.items():
        if label == 'all bins':
            tpv, fpv, fnv = tp_t, fp_t, fn_t
        else:
            tpv, fpv, fnv = tp_t[1:], fp_t[1:], fn_t[1:]
        slope, intercept = _fit_log_rho_trend(tpv, fpv, fnv, ts)

        # Pooled rho (P/R) over just the bins this variant fits on — differs
        # between variants because "excl. smallest bin" drops that bin's
        # TP/FP/FN from the pool entirely (not the same as global rho).
        tpv_sum, fpv_sum, fnv_sum = float(np.sum(tpv)), float(np.sum(fpv)), float(np.sum(fnv))
        p_v = tpv_sum / (tpv_sum + fpv_sum) if (tpv_sum + fpv_sum) > 0 else 0.0
        r_v = tpv_sum / (tpv_sum + fnv_sum) if (tpv_sum + fnv_sum) > 0 else 0.0
        rho_v = p_v / r_v if r_v > 0 else float('nan')

        variants.append({
            'label': variant_display_labels[label],
            'color': variant_colors[label],
            'slope': slope,
            'intercept': intercept,
            'rho': rho_v,
            'trend_result': boot_result['trends'][label],
        })

    _print_ci_outliers(obs_scores, obs_logrho, obs_names, variants, unit_label)

    _draw_rho_trend_plot(
        obs_scores, obs_logrho, obs_w, obs_names, variants,
        xlabel=xlabel,
        title=f'log(ρ) vs BBox Size trend ({unit_label}) — {modality} YOLOv11_{model_size}',
        fig_num=f"RhoTrend-{unit_label}-{modality}-{model_size}",
    )


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
        print(f"  Bin {bin_names[j]}: GT={num_gt[j]}, Det={num_detected[j]}, "
              f"P={precisions[j]:.3f}, R={recalls[j]:.3f}, F1={f1_scores[j]:.3f}, rho={rhos[j]:.3f}")
    print(f"Overall AP (px): {compute_ap(recalls, precisions):.4f}")
    px_scores = np.array([_bin_score(b, 10) for b in bin_names])
    print()

    # ── Bootstrap on rho (px bins) — secondary/diagnostic axis; note that   ──
    # ── mu_per_px differs across acquisition sources, so a "px bin" mixes   ──
    # ── physically different diameters if the test set spans several scopes ──
    _run_rho_trend_analysis(
        per_image_stats_px, bin_names, tp_arr, fp_arr, fn_arr, px_scores,
        unit_label='px', xlabel='GUV Size Range midpoint (pixels)',
        modality=modality, model_size=model_size,
    )

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
        print(f"  Bin {bin_names_um[j]} µm: GT={ngt_um[j]}, Det={ndet_um[j]}, "
              f"P={prec_um[j]:.3f}, R={rec_um[j]:.3f}, F1={f1_um[j]:.3f}, rho={rho_um[j]:.3f}")
    print(f"Overall AP (µm): {compute_ap(rec_um, prec_um):.4f}")
    um_scores = np.array([_bin_score(b, 5) for b in bin_names_um])
    print()

    # ── Bootstrap on rho (µm bins) — the physically meaningful axis, since   ──
    # ── mu_per_px differs across acquisition sources (Leica/Nikon high/low)  ──
    _run_rho_trend_analysis(
        per_image_stats_um, bin_names_um, tp_um, fp_um, fn_um, um_scores,
        unit_label='µm', xlabel='GUV Size Range midpoint (µm)',
        modality=modality, model_size=model_size,
    )

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