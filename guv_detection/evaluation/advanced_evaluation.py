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
MU_PER_PX    = 0.339   # µm per pixel (Nikon)

# ── Style constants ────────────────────────────────────────────────────────────
FS_LABEL  = 14
FS_TICK   = 13
FS_LEGEND = 12
FS_ANNOT  = 8
FIGSIZE   = (12, 5)

COLOR_P   = "#084594"   # dark blue  — Precision
COLOR_R   = "#4292c6"   # mid blue   — Recall
COLOR_F1  = "#9ecae1"   # light blue — F1
COLOR_BAR = "#1b7837"   # test green — bars

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


def _guv_dim_um(box):
    """Dataset formula: sqrt(max_dim² + min_dim²) / sqrt(2) * MU_PER_PX."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    max_dim, min_dim = max(w, h), min(w, h)
    return np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2) * MU_PER_PX


def _draw_eval_plot(bin_names, precisions, recalls, f1_scores,
                    num_detected, num_gt, xlabel, title, fig_num,
                    legend_outside=False):
    """Shared plotting routine for px and µm evaluation figures."""
    x        = np.arange(len(bin_names))
    width    = 0.35
    y_offset = 5

    fig, ax1 = plt.subplots(figsize=FIGSIZE, num=fig_num)

    # ── Metric lines ──────────────────────────────────────────────────────
    ax1.plot(x, precisions, marker='o', label='Precision', color=COLOR_P,  linewidth=1.8)
    ax1.plot(x, recalls,    marker='s', label='Recall',    color=COLOR_R,  linewidth=1.8)
    ax1.plot(x, f1_scores,  marker='^', label='F1 Score',  color=COLOR_F1, linewidth=1.8)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Precision / Recall / F1', fontsize=FS_LABEL)
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
                       color=COLOR_BAR, alpha=0.9, edgecolor='white', linewidth=0.5,
                       label='Detected GUV')
    bars_gt  = ax2.bar(x + width / 2, num_gt, width,
                       color=COLOR_BAR, alpha=0.45, edgecolor='white', linewidth=0.5,
                       hatch='//', label='GT GUV')

    ax2.set_ylim(0, 1200)
    ax2.set_ylabel('Number of GUV', fontsize=FS_LABEL, color=COLOR_BAR)
    ax2.tick_params(axis='y', labelsize=FS_TICK, colors=COLOR_BAR)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_edgecolor(COLOR_BAR)
    ax2.yaxis.grid(True, linestyle=':', linewidth=0.8, color=COLOR_BAR, alpha=0.3, zorder=0)
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

    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png")))
    image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    # pixel bins
    px_edges  = list(range(0, 110, 10))
    px_labels = [f"{px_edges[i]}-{px_edges[i+1]}" for i in range(len(px_edges) - 1)]
    px_last   = ">100"

    # µm bins  (formula: sqrt(max²+min²)/sqrt(2) * MU_PER_PX, Nikon 0.1205)
    um_edges  = list(range(0, 55, 5))
    um_labels = [f"{um_edges[i]}-{um_edges[i+1]}" for i in range(len(um_edges) - 1)]
    um_last   = ">50"

    stats = {'TP': 0, 'FP': 0, 'FN': 0}
    stats_per_bin    = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'GT_total': 0})
    stats_per_bin_um = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'GT_total': 0})
    all_gt_count = 0

    for img_name in image_names:
        img_path = os.path.join(image_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_name}.png")
        img = Image.open(img_path)
        img_w, img_h = img.size

        gt_path = os.path.join(label_dir, f"{img_name}.txt")
        pred_path = os.path.join(pred_dir, 'labels', f"{img_name}.txt")

        gt_boxes = read_gt_boxes(gt_path, img_w, img_h)
        pred_boxes = read_pred_boxes(pred_path, img_w, img_h)


        # GT bins (px and µm)
        for gb in gt_boxes:
            gb_w = gb[2] - gb[0]
            gb_h = gb[3] - gb[1]
            dim_px = (gb_w + gb_h) / 2
            dim_um = _guv_dim_um(gb)
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
            dim_um = _guv_dim_um(pb)
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
            else:
                stats['FP'] += 1
                stats_per_bin[p_bin]['FP']          += 1
                stats_per_bin_um[p_bin_um]['FP']    += 1

        # FN
        for i, gb in enumerate(gt_boxes):
            if i not in matched_gt:
                gb_w = gb[2] - gb[0]
                gb_h = gb[3] - gb[1]
                dim_px = (gb_w + gb_h) / 2
                dim_um = _guv_dim_um(gb)
                g_bin    = get_bin(dim_px, px_edges, px_labels, px_last)
                g_bin_um = get_bin(dim_um, um_edges, um_labels, um_last)
                stats['FN'] += 1
                stats_per_bin[g_bin]['FN']          += 1
                stats_per_bin_um[g_bin_um]['FN']    += 1


    def bin_sort_key(b):
        try:
            return int(b.split('-')[0])
        except Exception:
            return 9999

    def _compute_metrics(spb):
        all_bins = sorted(spb.keys(), key=bin_sort_key)
        prec, rec, f1s, n_det, n_gt = [], [], [], [], []
        for b in all_bins:
            tp = spb[b]['TP']; fp = spb[b]['FP']; fn = spb[b]['FN']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            prec.append(p); rec.append(r); f1s.append(f)
            n_det.append(tp + fp); n_gt.append(spb[b]['GT_total'])
        return all_bins, prec, rec, f1s, np.array(n_det), np.array(n_gt)

    def compute_ap(recall, precision):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x    = np.linspace(0, 1, 101)
        func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        return func(np.interp(x, mrec, mpre), x)

    # ── Pixel plot ────────────────────────────────────────────────────────
    bin_names, precisions, recalls, f1_scores, num_detected, num_gt = \
        _compute_metrics(stats_per_bin)

    print(f"Total GT: {sum(num_gt)}, Total Detected: {sum(num_detected)}\n")
    print("Bin-wise Evaluation (pixels):")
    for j in range(len(bin_names)):
        print(f"  Bin {bin_names[j]}: GT={num_gt[j]}, Det={num_detected[j]}, "
              f"P={precisions[j]:.3f}, R={recalls[j]:.3f}, F1={f1_scores[j]:.3f}")
    print(f"Overall AP (px): {compute_ap(recalls, precisions):.4f}\n")

    _draw_eval_plot(
        bin_names, precisions, recalls, f1_scores, num_detected, num_gt,
        xlabel='GUV Size Range (pixels)',
        title=f'Precision / Recall / F1 by BBox Size (px) — {modality} YOLOv11_{model_size}',
        fig_num=f"Eval-px-{modality}-{model_size}",
    )

    # ── µm plot ───────────────────────────────────────────────────────────
    bin_names_um, prec_um, rec_um, f1_um, ndet_um, ngt_um = \
        _compute_metrics(stats_per_bin_um)

    print("Bin-wise Evaluation (µm):")
    for j in range(len(bin_names_um)):
        print(f"  Bin {bin_names_um[j]} µm: GT={ngt_um[j]}, Det={ndet_um[j]}, "
              f"P={prec_um[j]:.3f}, R={rec_um[j]:.3f}, F1={f1_um[j]:.3f}")
    print(f"Overall AP (µm): {compute_ap(rec_um, prec_um):.4f}\n")

    _draw_eval_plot(
        bin_names_um, prec_um, rec_um, f1_um, ndet_um, ngt_um,
        xlabel='GUV Size Range (µm)',
        title=f'Precision / Recall / F1 by BBox Size (µm) — {modality} YOLOv11_{model_size}',
        fig_num=f"Eval-um-{modality}-{model_size}",
        legend_outside=True,
    )
            
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model on a folder of images")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/", 
                        help="Path to the folder containing images")
    args = parser.parse_args()

    folder = os.path.join(args.folder, f'DATA_training_{args.modality}_txt', 'test')
    evaluate(folder, args.model_size, args.modality)