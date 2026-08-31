"""
rho stability vs. GUV diameter, swept over conf_thresh x iou_thresh.

For every (conf_thresh, iou_thresh) operating point, fits the same
log(rho) = log(P) - log(R) vs. diameter trend used in advanced_evaluation.py
(bootstrap cluster-by-image, no other statistical machinery — see
Statistical_analysis.md), in two variants: using all qualifying µm bins, and
always excluding the 0-5µm bin (fixed physical exclusion, for comparability
across the whole grid — not "whichever bin is currently smallest",
unlike the single-cell version in advanced_evaluation.py). Only µm bins are
used; px bins mix physically different scales across acquisition sources
and are dropped from this analysis entirely.

Design record: Statistical_analysis.md (same folder). This is a v1
implementation — the sample-size-weighted confidence (per-cell alpha from
bootstrap CI width, §5.5 of the design doc) is a deliberately deferred
refinement, not implemented here. Cells are only ever fully opaque
(colored/gray) or hatched (insufficient data).

conf_thresh only sweeps upward from 0.25: the predbbox .txt files read here
were produced by model.predict() with its default conf=0.25 (see
tools/test.py), so nothing below that confidence was ever saved to disk —
sweeping below it would silently do nothing. iou_thresh cannot be swept by
re-thresholding a precomputed IoU matrix (the greedy matching assignment
itself depends on the threshold — see design doc §2 for the worked
counter-example), so every cell reruns the full matching pass.
"""

import argparse
import os
from glob import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from PIL import Image

from guv_detection.evaluation.advanced_evaluation import (
    _get_mu, iou, read_gt_boxes, read_pred_boxes, get_bin, _guv_dim_um,
    _bin_score, bootstrap_rho, MIN_N_BIN,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
)

# ── Sweep grid (tune freely — not yet settled, see design doc §7) ──────────
CONF_GRID  = np.round(np.arange(0.25, 0.91, 0.05), 2)     # upward from the 0.25 floor only
IOU_GRID   = np.round(np.arange(0.50, 0.96, 0.05), 2)     # standard COCO-style grid
N_BOOT_GRID = 2000                                        # reduced from the 2000 used for single-cell plots

# ── µm bins (same scheme as advanced_evaluation.py's evaluate()) ───────────
UM_EDGES  = list(range(0, 40, 5))
UM_LABELS = [f"{UM_EDGES[i]}-{UM_EDGES[i+1]}" for i in range(len(UM_EDGES) - 1)]
UM_LAST   = ">35"
ALL_UM_BIN_NAMES = UM_LABELS + [UM_LAST]
BIN_WIDTH_UM = 5
EXCLUDED_BIN_LABEL = UM_LABELS[0]   # "0-5" — always excluded in the second variant

STATE_CONSTANT     = 'constant'
STATE_SIGNIFICANT  = 'significant'
STATE_INSUFFICIENT = 'insufficient'

COLOR_SIGNIFICANT = "#9e9e9e"
CMAP_RHO = plt.get_cmap('Reds')   # monochromatic white -> red, low -> high rho

# Slightly smaller than advanced_evaluation.py's FS_LABEL/FS_TICK, just for this heatmap.
FS_LABEL_MATRIX = FS_LABEL - 2
FS_TICK_MATRIX  = FS_TICK - 2


def _load_images(folder, model_size):
    """Read GT boxes and ALL predictions (unfiltered by confidence, conf kept) once per image."""
    image_dir = os.path.join(folder, 'images')
    label_dir = os.path.join(folder, 'labels')
    pred_dir  = os.path.join(folder, f'predict_{model_size}')

    assert os.path.isdir(image_dir), f"Missing images folder: {image_dir}"
    assert os.path.isdir(label_dir), f"Missing labels folder: {label_dir}"
    assert os.path.isdir(pred_dir),  f"Missing predictions folder: {pred_dir}"

    is_leica = "leica" in folder.lower()

    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png")))
    image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    images = {}
    for img_name in image_names:
        img_path = os.path.join(image_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_name}.png")
        img_w, img_h = Image.open(img_path).size

        gt_path   = os.path.join(label_dir, f"{img_name}.txt")
        pred_path = os.path.join(pred_dir, 'labels', f"{img_name}.txt")

        images[img_name] = {
            'gt':   read_gt_boxes(gt_path, img_w, img_h),
            'pred': read_pred_boxes(pred_path, img_w, img_h, conf_thresh=0.0),
            'mu':   _get_mu(img_name, is_leica),
        }
    return images


def _match_image_cell(image_data, conf_thresh, iou_thresh):
    """
    Full greedy matching pass for one image at a fixed (conf_thresh,
    iou_thresh) cell, binned in µm only. Predictions are filtered by
    confidence and sorted by confidence descending before matching (highest
    confidence gets first claim on the best-overlapping GT box). Must be
    redone in full for every (conf, iou) combination — see module docstring.
    """
    gt_boxes = image_data['gt']
    mu = image_data['mu']
    preds = [p for p in image_data['pred'] if p[4] >= conf_thresh]
    preds.sort(key=lambda b: b[4], reverse=True)

    bin_counts = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    matched_gt = set()

    for pb in preds:
        dim_um = _guv_dim_um(pb, mu)
        p_bin  = get_bin(dim_um, UM_EDGES, UM_LABELS, UM_LAST)

        best_iou, best_gt = 0, -1
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou_val = iou(pb, gb)
            if iou_val > best_iou:
                best_iou, best_gt = iou_val, i

        if best_iou >= iou_thresh:
            matched_gt.add(best_gt)
            bin_counts[p_bin]['TP'] += 1
        else:
            bin_counts[p_bin]['FP'] += 1

    for i, gb in enumerate(gt_boxes):
        if i not in matched_gt:
            dim_um = _guv_dim_um(gb, mu)
            g_bin  = get_bin(dim_um, UM_EDGES, UM_LABELS, UM_LAST)
            bin_counts[g_bin]['FN'] += 1

    return bin_counts


def _run_cell(images, conf_thresh, iou_thresh):
    """Match every image at this (conf_thresh, iou_thresh) cell; pool per-bin and per-image counts."""
    stats_per_bin = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    per_image_stats = {}
    for img_name, data in images.items():
        counts = _match_image_cell(data, conf_thresh, iou_thresh)
        per_image_stats[img_name] = counts
        for b, c in counts.items():
            stats_per_bin[b]['TP'] += c['TP']
            stats_per_bin[b]['FP'] += c['FP']
            stats_per_bin[b]['FN'] += c['FN']
    return stats_per_bin, per_image_stats


def _pooled_rho(tp, fp, fn):
    """rho = P/R pooled over the given bins (same subset used by the trend fit — design doc §5.3)."""
    tp_s, fp_s, fn_s = float(np.sum(tp)), float(np.sum(fp)), float(np.sum(fn))
    if (tp_s + fp_s) <= 0 or (tp_s + fn_s) <= 0:
        return float('nan')
    p = tp_s / (tp_s + fp_s)
    r = tp_s / (tp_s + fn_s)
    return p / r if r > 0 else float('nan')


def _classify_cell(trend_entry, n_bins):
    """
    Three-state classification (design doc §5.4), WITHOUT the sample-size
    confidence weighting of §5.5 (deferred — see module docstring): a
    'constant' cell is always drawn fully opaque here, regardless of how
    little data supports that verdict.
    """
    if n_bins < 2 or trend_entry is None:
        return STATE_INSUFFICIENT
    s = trend_entry['slope']
    if s['n_valid'] == 0:
        return STATE_INSUFFICIENT
    excludes_zero = s['ci_lo'] > 0 or s['ci_hi'] < 0
    return STATE_SIGNIFICANT if excludes_zero else STATE_CONSTANT


def run_sweep(folder, model_size, conf_grid=CONF_GRID, iou_grid=IOU_GRID, n_boot=N_BOOT_GRID):
    """
    Sweep the full (conf_thresh, iou_thresh) grid. Returns
    {'all bins': {'rho': matrix, 'state': matrix}, 'excl 0-5um': {...}},
    each matrix shaped (len(conf_grid), len(iou_grid)).
    """
    images = _load_images(folder, model_size)
    scores_all = np.array([_bin_score(b, BIN_WIDTH_UM) for b in ALL_UM_BIN_NAMES])
    excl_mask_fixed = np.array([b != EXCLUDED_BIN_LABEL for b in ALL_UM_BIN_NAMES])

    n_conf, n_iou = len(conf_grid), len(iou_grid)
    matrices = {
        'all bins':   {'rho': np.full((n_conf, n_iou), np.nan),
                       'state': np.full((n_conf, n_iou), STATE_INSUFFICIENT, dtype=object)},
        'excl 0-5um': {'rho': np.full((n_conf, n_iou), np.nan),
                       'state': np.full((n_conf, n_iou), STATE_INSUFFICIENT, dtype=object)},
    }

    for i, conf_thresh in enumerate(conf_grid):
        for j, iou_thresh in enumerate(iou_grid):
            stats_per_bin, per_image_stats = _run_cell(images, conf_thresh, iou_thresh)

            tp_arr = np.array([stats_per_bin[b]['TP'] for b in ALL_UM_BIN_NAMES], dtype=float)
            fp_arr = np.array([stats_per_bin[b]['FP'] for b in ALL_UM_BIN_NAMES], dtype=float)
            fn_arr = np.array([stats_per_bin[b]['FN'] for b in ALL_UM_BIN_NAMES], dtype=float)

            qualifies = (tp_arr + fp_arr >= MIN_N_BIN) & (tp_arr + fn_arr >= MIN_N_BIN)

            trend_variants = {}
            cell_masks = {}
            if qualifies.sum() >= 2:
                trend_variants['all bins'] = (
                    [b for b, ok in zip(ALL_UM_BIN_NAMES, qualifies) if ok], scores_all[qualifies])
                cell_masks['all bins'] = qualifies

            qualifies_excl = qualifies & excl_mask_fixed
            if qualifies_excl.sum() >= 2:
                trend_variants['excl 0-5um'] = (
                    [b for b, ok in zip(ALL_UM_BIN_NAMES, qualifies_excl) if ok], scores_all[qualifies_excl])
                cell_masks['excl 0-5um'] = qualifies_excl

            if not trend_variants:
                continue   # both matrices keep their default 'insufficient' state / NaN rho

            boot_result = bootstrap_rho(per_image_stats, ALL_UM_BIN_NAMES, trend_variants, n_boot=n_boot)

            for label in ('all bins', 'excl 0-5um'):
                if label not in trend_variants:
                    continue
                mask = cell_masks[label]
                matrices[label]['rho'][i, j]   = _pooled_rho(tp_arr[mask], fp_arr[mask], fn_arr[mask])
                matrices[label]['state'][i, j] = _classify_cell(boot_result['trends'][label], int(mask.sum()))

        print(f"  conf_thresh={conf_thresh:.2f} done ({i + 1}/{n_conf})")

    return matrices


def _draw_rho_matrix(rho_matrix, state_matrix, conf_grid, iou_grid, title, fig_num):
    """Colored heatmap: colored=constant, gray=significant. Cells without enough
    data to fit a trend (insufficient-data state) are left blank, not drawn."""
    n_conf, n_iou = rho_matrix.shape
    fig, ax = plt.subplots(figsize=(0.55 * n_iou + 3.5, 0.55 * n_conf + 2.5), num=fig_num,
                            constrained_layout=True)

    finite_rho = rho_matrix[np.isfinite(rho_matrix)]
    if finite_rho.size > 0:
        vmin, vmax = float(np.min(finite_rho)), float(np.max(finite_rho))
    else:
        vmin, vmax = 0.0, 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, conf_thresh in enumerate(conf_grid):
        for j, iou_thresh in enumerate(iou_grid):
            state = state_matrix[i, j]
            rho_val = rho_matrix[i, j]

            if state == STATE_CONSTANT:
                facecolor = CMAP_RHO(norm(rho_val))
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=facecolor, edgecolor='white', linewidth=0.6))
                ax.text(j + 0.5, i + 0.5, f"{rho_val:.3f}", ha='center', va='center', fontsize=FS_ANNOT)
            elif state == STATE_SIGNIFICANT:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=COLOR_SIGNIFICANT, edgecolor='white', linewidth=0.6))
                ax.text(j + 0.5, i + 0.5, f"{rho_val:.3f}", ha='center', va='center', fontsize=FS_ANNOT, color='white')

    ax.set_xlim(0, n_iou)
    ax.set_ylim(0, n_conf)
    ax.set_xticks(np.arange(n_iou) + 0.5)
    ax.set_xticklabels([f"{v:.2f}" for v in iou_grid], rotation=45, fontsize=FS_TICK_MATRIX)
    ax.set_yticks(np.arange(n_conf) + 0.5)
    ax.set_yticklabels([f"{v:.2f}" for v in conf_grid], fontsize=FS_TICK_MATRIX)
    ax.set_xlabel('IoU threshold', fontsize=FS_LABEL_MATRIX)
    ax.set_ylabel('Confidence threshold', fontsize=FS_LABEL_MATRIX)

    legend_elements = [
        Patch(facecolor=CMAP_RHO(0.85), edgecolor='white', label='ρ constant (colored = ρ value)'),
        Patch(facecolor=COLOR_SIGNIFICANT, edgecolor='white', label='significant size-trend'),
    ]
    fig.legend(handles=legend_elements, fontsize=FS_LEGEND, loc='outside right upper')

    # plt.title(title, fontsize=FS_LABEL + 2)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep conf_thresh x iou_thresh and check whether rho (P/R) is stable across GUV diameter")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/", help="Path to the root folder (parent of DATA_training_*_txt)")
    parser.add_argument("--subfolder", type=str, default="test", help="Subfolder inside DATA_training_{modality}_txt, e.g. 'test' or 'Leica-tot/test'")
    args = parser.parse_args()

    folder = os.path.join(args.folder, f'DATA_training_{args.modality}_txt', args.subfolder)
    matrices = run_sweep(folder, args.model_size)

    for label, mats in matrices.items():
        _draw_rho_matrix(
            mats['rho'], mats['state'], CONF_GRID, IOU_GRID,
            title=f'ρ stability vs diameter — {label} — {args.modality} YOLOv11_{args.model_size}',
            fig_num=f"RhoMatrix-{label.replace(' ', '')}-{args.modality}-{args.model_size}",
        )
