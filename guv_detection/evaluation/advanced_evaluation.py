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

CONF_THRESH = 0.25
IOU_THRESH = 0.5

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

def get_bin(area, bin_edges, bin_labels):
    for i in range(len(bin_edges)-1):
        if bin_edges[i] <= area < bin_edges[i+1]:
            return bin_labels[i]
    return ">100"

def evaluate(folder, model_size, modality):
    image_dir = os.path.join(folder, 'images')
    label_dir = os.path.join(folder, 'labels')
    pred_dir  = os.path.join(folder, f'predict_{model_size}')

    assert os.path.isdir(image_dir), f"Missing images folder: {image_dir}"
    assert os.path.isdir(label_dir), f"Missing labels folder: {label_dir}"
    assert os.path.isdir(pred_dir),  f"Missing predictions folder: {pred_dir}"

    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png")))
    image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    bin_edges = list(range(0, 110, 10))  # fino a 60 incluso
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]

    stats = {'TP': 0, 'FP': 0, 'FN': 0}
    stats_per_bin = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'GT_total': 0})
    all_detections = []   # [(conf, TP/FP flag, bin_label)]
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


        # All guv GT in bins
        for gb in gt_boxes:
            gb_width = gb[2] - gb[0]
            gb_height = gb[3] - gb[1]
            guv_dimension = (gb_width + gb_height) / 2  # oppure (gb_width * gb_height)**0.5
            g_bin = get_bin(guv_dimension, bin_edges, bin_labels)
            stats_per_bin[g_bin]['GT_total'] += 1
            all_gt_count += 1

        matched_gt = set()
        for pb in pred_boxes:
            # Calcolo dimensione bbox predetto come media tra larghezza e altezza (oppure sqrt(area))
            p_width = pb[2] - pb[0]
            p_height = pb[3] - pb[1]
            guv_dimension = (p_width + p_height) / 2  # oppure: (p_width * p_height)**0.5
            p_bin = get_bin(guv_dimension, bin_edges, bin_labels)

            best_iou = 0
            best_gt = -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou_val = iou(pb, gb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = i

            conf = pb[4] if len(pb) > 4 else 1.0
            if best_iou >= IOU_THRESH:
                matched_gt.add(best_gt)
                stats['TP'] += 1
                stats_per_bin[p_bin]['TP'] += 1
                all_detections.append((conf, 1, p_bin))  # TP

            else:
                stats['FP'] += 1
                stats_per_bin[p_bin]['FP'] += 1
                all_detections.append((conf, 0, p_bin))  # FP
                

        # Calcolo FN per ogni ground truth non matchato, con bin in base alla dimensione GT
        for i, gb in enumerate(gt_boxes):
            if i not in matched_gt:
                gb_width = gb[2] - gb[0]
                gb_height = gb[3] - gb[1]
                guv_dimension = (gb_width + gb_height) / 2  # oppure: (gb_width * gb_height)**0.5
                g_bin = get_bin(guv_dimension, bin_edges, bin_labels)

                stats['FN'] += 1
                stats_per_bin[g_bin]['FN'] += 1


    # Ordinamento corretto dei bin
    def bin_sort_key(bin_name):
        try:
            start = int(bin_name.split('-')[0])
        except:
            start = 9999  # "60+" or others go last
        return start

    # Calcolo precision, recall e volumi
    bin_names = sorted(stats_per_bin.keys(), key=bin_sort_key)
    precisions, recalls, f1_scores = [], [], []
    num_detected, num_gt = [], []

    for b in bin_names:
        tp = stats_per_bin[b]['TP']
        fp = stats_per_bin[b]['FP']
        fn = stats_per_bin[b]['FN']
        gt_total = stats_per_bin[b]['GT_total']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        num_detected.append(tp + fp)
        num_gt.append(gt_total)

    # print the sum of num_gt and num_detected
    print(f"Total GT: {sum(num_gt)}, Total Detected: {sum(num_detected)}\n")

    print("Bin-wise Evaluation:")
    for jj in range(len(bin_names)):
        print(f"  Bin {bin_names[jj]}: GT={num_gt[jj]}, Detected={num_detected[jj]}, Precision={precisions[jj]:.3f}, Recall={recalls[jj]:.3f}, F1={f1_scores[jj]:.3f}")
    print()

    def compute_ap(recall, precision):
        """
        Compute the average precision (AP) using COCO-style 101-point interpolation.
        (Identical to Ultralytics' implementation)
        """
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Precision envelope (monotone decreasing)
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integration method (COCO 101-point)
        x = np.linspace(0, 1, 101)
        func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        ap = func(np.interp(x, mrec, mpre), x)

        return ap, mpre, mrec

    ap, mpre, mrec = compute_ap(recalls, precisions)
    print(f"Overall AP: {ap:.4f}\n")


    ## PLOTTING
    x = np.arange(len(bin_names))
    width = 0.35

    # Plot
    fig, ax1 = plt.subplots(figsize=(16, 8), num=f"Evaluation-{modality}-YOLOv11_{model_size}")

    # Precision & Recall lines
    ax1.plot(x, precisions, marker='o', label='Precision', color='blue')
    ax1.plot(x, recalls, marker='o', label='Recall', color='green')
    ax1.plot(x, f1_scores, marker='o', label='F1 Score', color='orange')
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Precision / Recall', color='black', fontsize=20)
    ax1.set_xlabel('BBox Size Range (pixels)', color='black', fontsize=22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_names, rotation=45, fontsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.grid(linestyle=':', alpha=0.7)

    # Secondary axis for counts
    ax2 = ax1.twinx()
    ax2.bar(x - width/2, num_detected, width, label='Detected GUV', color='skyblue', alpha=0.6)
    ax2.bar(x + width/2, num_gt, width, label='GUV', color='gray', alpha=0.5)
    ax2.set_ylabel('Number of GUV', color='black', fontsize=20)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.grid(linestyle=':', alpha=0.7)

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=18)

    plt.title('Precision/Recall and Object Counts by BBox Size Range', fontsize=26)
    plt.tight_layout()
    plt.show()
            
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model on a folder of images")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/", 
                        help="Path to the folder containing images")
    args = parser.parse_args()

    folder = os.path.join(args.folder, f'DATA_training_{args.modality}_txt', 'test')
    evaluate(folder, args.model_size, args.modality)