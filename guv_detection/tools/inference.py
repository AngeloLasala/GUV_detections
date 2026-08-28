"""
Run GUV detection inference on a folder of images and extrapolate the
size distribution (histogram, log-normal fit, CSV).
"""
import os
import csv
import argparse
import ultralytics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import lognorm


def read_pred_boxes(pred_path, conf_thresh):
    """Read predicted boxes with confidence filtering."""
    boxes = []
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls, xc, yc, w, h, conf = map(float, parts[:6])
                    if conf >= conf_thresh:
                        boxes.append([xc, yc, w, h, conf])
    return boxes


def main(args):
    """Run inference on a folder of images and extrapolate GUV size distribution."""

    model = ultralytics.YOLO(args.model)
    model.predict(source=args.folder, save=True, save_txt=True, save_conf=True,
                  project=args.folder)

    prediction_folder = os.path.join(args.folder, 'predict', 'labels')
    if not os.path.exists(prediction_folder):
        print("No predictions found. Please check the model and input folder.")
        return

    count_edge = 0
    count_inter = 0
    dim_list = []
    for label_file in os.listdir(prediction_folder):
        if not label_file.endswith('.txt'):
            continue
        pred_path = os.path.join(prediction_folder, label_file)
        image_name = os.path.join(args.folder, 'predict', label_file.replace('.txt', '.jpg'))
        w_img, h_img = Image.open(image_name).size
        for xc, yc, wbox, hbox, conf in read_pred_boxes(pred_path, args.conf_thresh):
            W = wbox * w_img
            H = hbox * h_img
            max_dim = max(W, H)
            min_dim = min(W, H)
            if min_dim <= 0.5 * max_dim:
                count_edge += 1
            else:
                count_inter += 1
            dim = np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2)
            dim_list.append(dim * args.mu_per_pixel)

    if not dim_list:
        print("No GUVs detected across all images.")
        return

    arr = np.array(dim_list)
    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    shape, loc, scale = lognorm.fit(arr, floc=0)
    mu_ln = float(np.log(scale))
    sigma_ln = float(shape)

    print(f"Total GUVs detected: {len(dim_list)}")
    print(f"Edge GUVs: {count_edge}, Interior GUVs: {count_inter}")
    print(f"Median: {median:.2f} um  Q1: {q1:.2f} um  Q3: {q3:.2f} um")
    print(f"Log-normal fit: mu={mu_ln:.2f}, sigma={sigma_ln:.2f}")

    # CSV
    csv_path = os.path.join(args.folder, 'GUV_size_distribution.csv')
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GUV_ID", "Diameter_um"])
        for i, diameter in enumerate(dim_list, start=1):
            writer.writerow([i, f"{diameter:.4f}"])

    # Histogram + log-normal fit
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), tight_layout=True)
    bin_width = 5
    bins = np.arange(0, max(dim_list) + bin_width, bin_width)
    ax.hist(dim_list, bins=bins, color='chocolate', alpha=0.5, density=True)
    ax.axvline(median, color='darkred', linestyle='dashed', linewidth=3, label=f'Median: {median:.2f} um')
    ax.axvline(q1, color='red', linestyle='dashed', linewidth=3, label=f'Q1: {q1:.2f} um')
    ax.axvline(q3, color='red', linestyle='dashed', linewidth=3, label=f'Q3: {q3:.2f} um')
    x = np.linspace(0, max(dim_list), 1000)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
    ax.plot(x, pdf, 'k-', linewidth=3,
            label=f'Log-normal fit\nmu={mu_ln:.2f}, sigma={sigma_ln:.2f}\nTotal GUVs: {len(dim_list)}')
    ax.set_xlabel('GUV Diameter (um)', fontsize=20)
    ax.set_ylabel('Density of GUVs', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(linestyle=':')

    plot_path = os.path.join(args.folder, 'GUV_size_distribution.pdf')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    print(f"CSV saved: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GUV detection inference on a folder of images")
    parser.add_argument("--model", type=str, default="/home/angelo/Documenti/GUV_detector/train_rgb/weights/best.pt",
                         help="Path to the model file")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_txt/inference",
                         help="Path to the folder containing images")
    parser.add_argument("--mu_per_pixel", type=float, default=0.339, help="Conversion factor from pixels to micrometers")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold for predictions")
    args = parser.parse_args()

    main(args)
