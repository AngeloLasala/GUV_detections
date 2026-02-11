"""
Extrapolate the dimention of GUV and ather information
"""
import os
import argparse
import ultralytics
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    """
    Main function. Performe the analysis across the folder of images
    """

    ## read the model
    model = ultralytics.YOLO(args.model)

    ## model prediction on the folder
    results = model.predict(source=args.folder, save=True, save_txt=True, save_conf=True,
                            project=args.folder)

    # check if the predict formde is create
    if not os.path.exists(os.path.join(args.folder, 'predict')):
        print("No predictions found. Please check the model and input folder.")
        return None
    
    prediction_folder = os.path.join(args.folder, 'predict', 'labels')

    count_edge = 0
    count_inter = 0
    dim_list = []
    for i in os.listdir(prediction_folder):
        if i.endswith('.txt'):
            pred_path = os.path.join(prediction_folder, i)
            image_name = os.path.join(args.folder, 'predict', i.replace('.txt', '.jpg'))
            w, h = Image.open(image_name).size
            boxes = read_pred_boxes(pred_path, args.conf_thresh)
            for bbox in boxes:
                w_guv = bbox[2] * w 
                h_guv = bbox[3] * h

                max_dim = max(w_guv, h_guv)
                min_dim = min(w_guv, h_guv)

                if min_dim <= 0.5 * max_dim:
                    count_edge += 1
                else:
                    count_inter += 1
                
                dim = max_dim * args.mu_per_pixel
                dim_list.append(dim)
    
    print(f"Total GUVs detected: {len(dim_list)}")
    print(f"Edge GUVs: {count_edge}, Inter GUVs: {count_inter}")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), tight_layout=True)
    bin_width = 5
    bins = np.arange(0, max(dim_list) + bin_width, bin_width)
    ax.hist(dim_list, bins=bins, color='chocolate', alpha=0.5)
    median = np.median(dim_list)
    first_quartile = np.percentile(dim_list, 25)
    third_quartile = np.percentile(dim_list, 75)
    ax.axvline(median, color='darkred', linestyle='dashed', linewidth=3, label=f'Median: {median:.2f} μm')
    ax.axvline(first_quartile, color='red', linestyle='dashed', linewidth=3, label=f'Q1: {first_quartile:.2f} μm')
    ax.axvline(third_quartile, color='red', linestyle='dashed', linewidth=3, label=f'Q3: {third_quartile:.2f} μm')

    # ymin, ymax = ax.get_ylim()
    # ax.fill_betweenx([ymin, ymax], first_quartile, third_quartile, color='orange', alpha=0.2)
    # ax.fill_betweenx([0, max(np.histogram(dim_list, bins=30)[0])], first_quartile, third_quartile, color='orange', alpha=0.2)

    # set x label
    ax.set_xlabel('GUV Diameter (μm)', fontsize=24)
    ax.set_ylabel('number of GUVs', fontsize=24)
    # set the tixk font size
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    # set the grid
    ax.grid(linestyle=':')
    plt.savefig(os.path.join(args.folder, 'GUV_size_distribution.pdf'), dpi=300)
    plt.show()
                

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Test a model on a folder of images")
    parser.add_argument("--model", type=str, default="/home/angelo/Documenti/GUV_detector/train_rgb/weights/best.pt",
                                             help="Path to the model file")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_txt/inference", 
                        help="Path to the folder containing images")
    parser.add_argument("--mu_per_pixel", type=float, default=0.339, help="Conversion factor from pixels to micrometers")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold for predictions")
    args = parser.parse_args()

    main(args)