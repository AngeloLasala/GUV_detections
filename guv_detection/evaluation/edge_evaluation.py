"""
Compute the advanced statistic evaluation about the effect of cutting
- edges detection
- effect on the size distribuction
"""
import argparse
import os
from glob import glob
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


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

def counting_edge_boxes(prediction_folder, mu_per_pixel, conf_thresh):
    """
    Counting the number of edge boxes in a image
    """

    # unique list of name
    names = []
    for i in os.listdir(prediction_folder):
        if i.endswith('.txt'):
            name = os.path.basename(i).split('.')[0].split('_')[:-1]
            name = "_".join(name)
            names.append(name)
    names = list(set(names))

    count_edge = 0
    count_inter = 0
    dict_name = {name: {} for name in names}
    dim_list = []
    for i in os.listdir(prediction_folder):
        if i.endswith('.txt'):
            ## get image and prediction
            pred_path = os.path.join(prediction_folder, i)
            image_name = os.path.join(args.folder, 'predict', i.replace('.txt', '.jpg'))
            cutting_id = os.path.basename(image_name).split('.')[0].split('_')[-1]
            name = os.path.basename(image_name).split('.')[0].split('_')[:-1]
            name = "_".join(name)

            ## read the bbox
            W, H = Image.open(image_name).size
            image_guv = np.array(Image.open(image_name))
            boxes = read_pred_boxes(pred_path, conf_thresh)
            # print(f"Image: {image_name}, GUVs detected: {len(boxes)}")
            toy_image = np.zeros((H, W), dtype=np.uint8)

            dict_name[name][cutting_id] = []

            for bbox in boxes:
                xc, yc, w, h, conf = bbox
                w_guv = w * W
                h_guv = h * H

                max_dim = max(w_guv, h_guv)
                min_dim = min(w_guv, h_guv)

                if ((xc <= 0.05) or (xc >= 0.95) or (yc <= 0.05) or (yc  >= 0.95)) and (min_dim <= 0.75 * max_dim):
                # if min_dim <= 0.5 * max_dim:
                    # print(f"Edge box detected: xc={xc}, yc={yc}, w={w}, h={h}, conf={conf}")
                    # draw the box on the image
                    x1 = int((xc - w/2) * W)
                    y1 = int((yc - h/2) * H)
                    x2 = int((xc + w/2) * W)
                    y2 = int((yc + h/2) * H)
                    dict_name[name][cutting_id].append([xc, yc, w, h, conf])
                    toy_image[y1:y2, x1:x2] = 255
                    count_edge += 1

                # internal boxes
                else:
                    count_inter += 1

                dim = max_dim * mu_per_pixel
                dim_list.append(dim)


            # fig, ax = plt.subplots(1, 2, figsize=(20, 6), tight_layout=True)
            # ax[0].imshow(image_guv, cmap='gray')
            # ax[0].set_title('Original Image')
            # ax[1].imshow(toy_image, cmap='gray')
            # ax[1].set_title('Edge Boxes Detected')
            # plt.show()

    print(f"Total GUVs detected: {len(dim_list)}")
    print(f"Edge GUVs: {count_edge}, Inter GUVs: {count_inter}\n")
    return dict_name, dim_list

def analysis_edge(dict_name):
    """
    Analysis of the cutting effect of unique identifier image
    """
    cutting_list = ['A1', 'A2', 'A3', 'A4',
                    'B1', 'B2', 'B3', 'B4',
                    'C1', 'C2', 'C3', 'C4',
                    'D1', 'D2', 'D3', 'D4']

    outer_edge_list = ['A1', 'A2', 'A3', 'A4',
                        'B1', 'B4',
                       'C1', 'C4',
                       'D1', 'D2', 'D3', 'D4']

    ## outer edge counting
    outer_dict = {}
    outer_count = 0
    edge_conditions = {
    "A1": lambda xc, yc: (xc <= 0.05) or (yc <= 0.05),
    "A2": lambda xc, yc: yc <= 0.05,
    "A3": lambda xc, yc: yc <= 0.05,  
    "A4": lambda xc, yc: (xc >= 0.95) or (yc <= 0.05),
    "B1": lambda xc, yc: xc <= 0.05,
    "B4": lambda xc, yc: xc >= 0.95,
    "C1": lambda xc, yc: xc <= 0.05,
    "C4": lambda xc, yc: xc >= 0.95,
    "D1": lambda xc, yc: (xc <= 0.05) or (yc >= 0.95),
    "D2": lambda xc, yc: yc >= 0.95,
    "D3": lambda xc, yc: yc >= 0.95,
    "D4": lambda xc, yc: (xc >= 0.95) or (yc >= 0.95),
    }
    outer_dict = {unique_image: {cut_id: [] for cut_id in outer_edge_list} for unique_image in dict_name.keys()}
    for unique_image, cuts in dict_name.items():
        for cut_id, guvs in cuts.items():
            if cut_id in edge_conditions:  # consideriamo solo i bordi
                for guv in guvs:
                    xc, yc, w, h, conf = guv
                    if edge_conditions[cut_id](xc, yc):
                        outer_dict[unique_image][cut_id].append(guv)
                        outer_count += 1
    print(f"Total outer edge GUVs detected: {outer_count}\n")
   

    ## inner edge counting
    inner_count = 0
    inner_conditions = {
        "A1": lambda xc, yc: (xc > 0.95) or (yc > 0.95),
        "A2": lambda xc, yc: (xc < 0.05) or (xc > 0.95 ) or (yc > 0.95),
        "A3": lambda xc, yc: (xc < 0.05) or (xc > 0.95 ) or (yc > 0.95),
        "A4": lambda xc, yc: (xc < 0.05) or (yc > 0.95),
        "B1": lambda xc, yc: (xc > 0.95) or (yc < 0.05) or (yc > 0.95),
        "B2": lambda xc, yc: (xc < 0.05) or (xc > 0.95) or (yc < 0.05) or (yc > 0.95),
        "B3": lambda xc, yc: (xc < 0.05) or (xc > 0.95) or (yc < 0.05) or (yc > 0.95),
        "B4": lambda xc, yc: (xc < 0.05) or (yc < 0.05) or (yc > 0.95),
        "C1": lambda xc, yc: (xc > 0.95) or (yc < 0.05) or (yc > 0.95),
        "C2": lambda xc, yc: (xc < 0.05) or (xc > 0.95) or (yc < 0.05) or (yc > 0.95),
        "C3": lambda xc, yc: (xc < 0.05) or (xc > 0.95) or (yc < 0.05) or (yc > 0.95),
        "C4": lambda xc, yc: (xc < 0.05) or (yc < 0.05) or (yc > 0.95),
        "D1": lambda xc, yc: (xc > 0.95) or (yc < 0.05),
        "D2": lambda xc, yc: (xc < 0.05) or (xc > 0.95) or (yc < 0.05),
        "D3": lambda xc, yc: (xc < 0.05) or (xc > 0.95) or (yc < 0.05),
        "D4": lambda xc, yc: (xc < 0.05) or (yc < 0.05)}

    inner_dict = {unique_image: {cut_id: [] for cut_id in cutting_list} for unique_image in dict_name.keys()}
    for unique_image, cuts in dict_name.items():
        for cut_id, guvs in cuts.items():
            if cut_id in inner_conditions:
                for guv in guvs:
                    xc, yc, w, h, conf = guv
                    if inner_conditions[cut_id](xc, yc):
                        inner_dict[unique_image][cut_id].append(guv)
                        inner_count += 1
    print(f"Total inner edge GUVs detected: {inner_count}\n")

    return outer_dict, inner_dict

def reconstruct_edge_map(outer_dict, inner_dict, prediction_folder):
    """
    recostructed edge map
    """

    cut_id_positions = {
            'A1': (0, 0), 'A2': (0, 1), 'A3': (0, 2), 'A4': (0, 3),
            'B1': (1, 0), 'B2': (1, 1), 'B3': (1, 2), 'B4': (1, 3),
            'C1': (2, 0), 'C2': (2, 1), 'C3': (2, 2), 'C4': (2, 3),
            'D1': (3, 0), 'D2': (3, 1), 'D3': (3, 2), 'D4': (3, 3)
        }
    H, W = 542, 1024  # Dimensioni fisse per le immagini
    
    for unique_image in outer_dict.keys():
        print(f"Unique image: {unique_image}")
        fig, ax = plt.subplots(4, 4, figsize=(17, 10), tight_layout=True, num=unique_image)
        for cut_id in outer_dict[unique_image].keys():
            combined_image = np.zeros((H, W, 3), dtype=np.uint8)
            for guv in outer_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv
                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)
                combined_image[y1:y2, x1:x2, 0] = 255

            for guv in inner_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv
                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)
                combined_image[y1:y2, x1:x2, 1] = 255
            pos = cut_id_positions[cut_id]
            ax[pos].imshow(combined_image)
            ax[pos].set_title(f'{cut_id}', fontsize=16)
            ax[pos].axis('off')

        for cut_id in ['B2', 'B3', 'C2', 'C3']:
            combined_image = np.zeros((H, W, 3), dtype=np.uint8)
            for guv in inner_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv
                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)
                combined_image[y1:y2, x1:x2, 1] = 255
            pos = cut_id_positions[cut_id]
            ax[pos].imshow(combined_image)
            ax[pos].set_title(f'{cut_id}', fontsize=16)
            ax[pos].axis('off')


        ## recostrctuted image
        fig1, ax1 = plt.subplots(4, 4, figsize=(17, 10), tight_layout=True, num=unique_image+"_real")
        for cut_id in cut_id_positions.keys():
            ## read the image
            image_name = os.path.join(prediction_folder, unique_image + f"_{cut_id}.jpg")
            print(f"Reading image: {image_name}")
            if os.path.exists(image_name):
                image = np.array(Image.open(image_name).convert("RGB"))
            else:
                image = np.zeros((H, W, 3), dtype=np.uint8)
            pos = cut_id_positions[cut_id]
            ax1[pos].imshow(image)
            ax1[pos].set_title(f'{cut_id}', fontsize=16)
            ax1[pos].axis('off')
        plt.show()

                





def main(args):
    """
    Compute the statistic about the cutting effect
    """
    ## read images and predictions
    folder = args.folder
    image_dir = os.path.join(folder)
    pred_dir  = os.path.join(folder, 'predict')
    assert os.path.isdir(pred_dir),  f"Missing predictions folder: {pred_dir}"

    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png")))
    image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    prediction_folder = os.path.join(pred_dir, 'labels')

    dict_edges, dim_edge_list = counting_edge_boxes(prediction_folder, args.mu_per_pixel, args.conf_thresh)

    outer_dict, inner_dict = analysis_edge(dict_edges)

    reconstruct_edge_map(outer_dict, inner_dict, pred_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced evaluation script for object detection models using YOLO format.")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_txt/inference",
                        help="Path to the folder containing images")
    parser.add_argument("--mu_per_pixel", type=float, default=0.3339, help="Conversion factor from pixels to micrometers")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold for predictions")
    args = parser.parse_args()

    main(args)


