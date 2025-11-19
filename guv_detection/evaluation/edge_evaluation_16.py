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
import copy


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
    central_name = {name: {} for name in names}
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
            central_name[name][cutting_id] = []

            for bbox in boxes:
                xc, yc, w, h, conf = bbox
                w_guv = w * W
                h_guv = h * H

                max_dim = max(w_guv, h_guv)
                min_dim = min(w_guv, h_guv)

                if ((xc <= 0.10) or (xc >= 0.90) or (yc <= 0.10) or (yc  >= 0.90)) and (min_dim <= 0.85 * max_dim):
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
                    x1 = int((xc - w/2) * W)
                    y1 = int((yc - h/2) * H)
                    x2 = int((xc + w/2) * W)
                    y2 = int((yc + h/2) * H)
                    toy_image[y1:y2, x1:x2] = 255
                    central_name[name][cutting_id].append([xc, yc, w, h, conf])
                    count_inter += 1

                dim = max_dim * mu_per_pixel
                dim_list.append(dim)

    print(f"Total GUVs detected: {len(dim_list)}")
    print(f"Edge GUVs: {count_edge}, Inter GUVs: {count_inter}\n")
    return dict_name, central_name, dim_list

def analysis_edge(dict_name, central_name):
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
                for xx, guv in enumerate(guvs):
                    xc, yc, w, h, conf = guv
                    if edge_conditions[cut_id](xc, yc):
                        outer_dict[unique_image][cut_id].append(guv)
                        outer_count += 1
    print(f"Total outer edge GUVs detected: {outer_count}")
   
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
            # if cut_id in inner_conditions:
            for yy, guv in enumerate(guvs):
                xc, yc, w, h, conf = guv
                # if inner_conditions[cut_id](xc, yc):
                inner_dict[unique_image][cut_id].append(guv)
                inner_count += 1
    print(f"Total inner edge GUVs detected: {inner_count}")
    
    center_dict = {unique_image: {cut_id: [] for cut_id in cutting_list} for unique_image in central_name.keys()}  
    center_count = 0
    for unique_image, cuts in central_name.items():
        for cut_id, guvs in cuts.items():
            for guv in guvs:
                xc, yc, w, h, conf = guv
                center_dict[unique_image][cut_id].append(guv)
                center_count += 1 
    print(f"Total central GUVs detected: {center_count}\n")

    return outer_dict, inner_dict, center_dict

def counting_cutted_guv(outer_dict, inner_dict, prediction_folder, conf_thresh, mu_per_pixel):
    """
    """
    anchor_cut = ['A2', 'A4', 
                  'B1', 'B2', 'B3', 
                  'C2', 'C3', 'C4',
                  'D1', 'D3']

    adiacent_cut = {'A2': ['A1', 'A3'],
                    'A4': ['A3', 'B4'],
                    'B1': ['A1', 'C1'],
                    'B2': ['A2', 'B1', 'B3', 'C2'],
                    'B3': ['A3', 'B4', 'C3'],
                    'C2': ['C1', 'C3', 'D2'],
                    'C3': ['C4', 'D3'],
                    'C4': ['B4', 'D4'],
                    'D1': ['C1', 'D2'],
                    'D3': ['D2', 'D4']}

    new_inner_dict = copy.deepcopy(inner_dict)
    for unique_image in inner_dict.keys():
        print(f"Unique image: {unique_image}")
        for cut_id in anchor_cut:
            if len(inner_dict[unique_image][cut_id]) > 0:
                for cut_id_adjacent in adiacent_cut[cut_id]:
                    if len(inner_dict[unique_image][cut_id_adjacent]) > 0:
                        # print(f"  Anchor {cut_id}: {len(inner_dict[unique_image][cut_id])} -  Adjacent {cut_id_adjacent}: {len(inner_dict[unique_image][cut_id_adjacent])}")
                        
                        for ii, guv_ancor in enumerate(inner_dict[unique_image][cut_id]):
                            xc_anc, yc_anc, _, _, _ = guv_ancor
                            for guv_adj in inner_dict[unique_image][cut_id_adjacent]:
                                xc_adj, yc_adj, _, _, _ = guv_adj

                                if cut_id == 'A2' and cut_id_adjacent == 'A1':
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs(xc_anc - (1-xc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                        print(inner_dict[unique_image][cut_id][ii][4], new_inner_dict[unique_image][cut_id][ii][4])

                                if cut_id == 'A2' and (cut_id_adjacent == 'A3'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                               
                                if cut_id == 'A4' and (cut_id_adjacent == 'A3'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'A4' and (cut_id_adjacent == 'B4'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs((1-yc_anc) - yc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'B1' and (cut_id_adjacent == 'A1'):# or cut_id_adjacent == 'C1'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs((1-yc_anc) - yc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'B1' and (cut_id_adjacent == 'C1'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs(yc_anc - (1-yc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'B2' and (cut_id_adjacent == 'B1'):#: or cut_id_adjacent == 'B3'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs(xc_anc - (1-xc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'B2' and (cut_id_adjacent == 'B3'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'B2' and (cut_id_adjacent == 'A2'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs(yc_anc - (1-yc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'B2' and (cut_id_adjacent == 'C2'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs((1-yc_anc) - yc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'B3' and cut_id_adjacent == 'A3':# or cut_id_adjacent == 'C3'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs(yc_anc - (1-yc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'B3' and (cut_id_adjacent == 'C3'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs((1-yc_anc) - yc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'B3' and (cut_id_adjacent == 'B4'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'C2' and (cut_id_adjacent == 'C1'):# or cut_id_adjacent == 'C3'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs(xc_anc - (1-xc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'C2' and (cut_id_adjacent == 'C3'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV  {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'C3' and (cut_id_adjacent == 'C4'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV  {cut_id_adjacent} at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'C3' and (cut_id_adjacent == 'D3'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs((1-yc_anc) - yc_adj) < 0.025:
                                        print(f" {cut_id}   Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                
                                if cut_id == 'C4' and (cut_id_adjacent == 'B4'):# or cut_id_adjacent == 'D4'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs(yc_anc - (1-yc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id  == 'C4' and (cut_id_adjacent == 'D4'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs((1-yc_anc) - yc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'D1' and (cut_id_adjacent == 'C1'):
                                    if np.abs(xc_anc - xc_adj) < 0.025 and np.abs(yc_anc - (1-yc_adj)) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
                                if cut_id == 'D1' and (cut_id_adjacent == 'D2'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f" {cut_id}   Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'D3' and (cut_id_adjacent == 'D4'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs((1-xc_anc) - xc_adj) < 0.025:
                                        print(f"  {cut_id}  Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0

                                if cut_id == 'D3' and (cut_id_adjacent == 'D2'):
                                    if np.abs(yc_anc - yc_adj) < 0.025 and np.abs(xc_anc - (1-xc_adj)) < 0.025:
                                        print(f" {cut_id}   Anchor GUV at ({xc_anc:.2f}, {yc_anc:.2f}) - Adjacent GUV at ({xc_adj:.2f}, {yc_adj:.2f})")
                                        new_inner_dict[unique_image][cut_id][ii][4] = 0.0
        print()
        
    return inner_dict, new_inner_dict
    

def reconstruct_edge_map(outer_dict, inner_dict, central_dict, prediction_folder, conf_thresh, mu_per_pixel):
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
    
    rms_list = []
    for unique_image in outer_dict.keys():
        # fig, ax = plt.subplots(4, 4, figsize=(17, 10), tight_layout=True, num=unique_image)
        for cut_id in outer_dict[unique_image].keys():
            combined_image = np.zeros((H, W, 3), dtype=np.uint8)
            for guv in outer_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv
                if conf > conf_thresh:
                    x1 = int((xc - w/2) * W)
                    y1 = int((yc - h/2) * H)
                    x2 = int((xc + w/2) * W)
                    y2 = int((yc + h/2) * H)

                    w_guv = w * W
                    h_guv = h * H
                    max_dim = max(w_guv, h_guv)
                    min_dim = min(w_guv, h_guv)
                    dim_rms = np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2) * mu_per_pixel
                    # rms_list.append(dim_rms)
                
                    # combined_image[y1:y2, x1:x2, 0] = 255
            

            for guv in inner_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv
                if conf > conf_thresh:
                    x1 = int((xc - w/2) * W)
                    y1 = int((yc - h/2) * H)
                    x2 = int((xc + w/2) * W)
                    y2 = int((yc + h/2) * H)

                    w_guv = w * W
                    h_guv = h * H
                    max_dim = max(w_guv, h_guv)
                    min_dim = min(w_guv, h_guv)
                    dim_rms = np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2) * mu_per_pixel 
                    rms_list.append(dim_rms)

                    combined_image[y1:y2, x1:x2, 1] = 255

            for guv in central_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv
                if conf > conf_thresh:
                    x1 = int((xc - w/2) * W)
                    y1 = int((yc - h/2) * H)
                    x2 = int((xc + w/2) * W)
                    y2 = int((yc + h/2) * H)

                    w_guv = w * W
                    h_guv = h * H
                    max_dim = max(w_guv, h_guv)
                    min_dim = min(w_guv, h_guv)
                    dim_rms = np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2) * mu_per_pixel 
                    rms_list.append(dim_rms)

                    combined_image[y1:y2, x1:x2, 2] = 150

            pos = cut_id_positions[cut_id]
            # ax[pos].imshow(combined_image)
            # ax[pos].set_title(f'{cut_id}', fontsize=16)
            # ax[pos].axis('off')

        for cut_id in ['B2', 'B3', 'C2', 'C3']:
            combined_image = np.zeros((H, W, 3), dtype=np.uint8)
            for guv in inner_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv      
                if conf > conf_thresh:
                    x1 = int((xc - w/2) * W)
                    y1 = int((yc - h/2) * H)
                    x2 = int((xc + w/2) * W)
                    y2 = int((yc + h/2) * H)

                    w_guv = w * W
                    h_guv = h * H
                    max_dim = max(w_guv, h_guv)
                    min_dim = min(w_guv, h_guv)
                    dim_rms = np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2) * mu_per_pixel                
                    rms_list.append(dim_rms)

                    combined_image[y1:y2, x1:x2, 1] = 255

            for guv in central_dict[unique_image][cut_id]:
                xc, yc, w, h, conf = guv
                if conf > conf_thresh:
                    x1 = int((xc - w/2) * W)
                    y1 = int((yc - h/2) * H)
                    x2 = int((xc + w/2) * W)
                    y2 = int((yc + h/2) * H)

                    w_guv = w * W
                    h_guv = h * H
                    max_dim = max(w_guv, h_guv)
                    min_dim = min(w_guv, h_guv)
                    dim_rms = np.sqrt(max_dim**2 + min_dim**2) / np.sqrt(2) * mu_per_pixel                
                    rms_list.append(dim_rms)

                    combined_image[y1:y2, x1:x2, 2] = 150
            pos = cut_id_positions[cut_id]
            # ax[pos].imshow(combined_image)
            # ax[pos].set_title(f'{cut_id}', fontsize=16)
            # ax[pos].axis('off')


        ## recostrctuted image
        # fig1, ax1 = plt.subplots(4, 4, figsize=(17, 10), tight_layout=True, num=unique_image+"_real")
        pred_guv = 0
        for cut_id in cut_id_positions.keys():
            ## read the image
            image_name = os.path.join(prediction_folder, unique_image + f"_{cut_id}.jpg")
            pred_path = os.path.join(prediction_folder, 'labels', unique_image + f"_{cut_id}.txt")
            pred_boxes = read_pred_boxes(pred_path, conf_thresh)
            pred_guv += len(pred_boxes)

            if os.path.exists(image_name):
                image = np.array(Image.open(image_name).convert("RGB"))
            else:
                image = np.zeros((H, W, 3), dtype=np.uint8)
            pos = cut_id_positions[cut_id]
        #     ax1[pos].imshow(image)
        #     ax1[pos].set_title(f'{cut_id}', fontsize=16)
        #     ax1[pos].axis('off')
        # plt.show()
        print(f"Unique image: {unique_image}, Predicted GUVs: {pred_guv}")
    
    return rms_list



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

    dict_edges, central_name, dim_edge_list = counting_edge_boxes(prediction_folder, args.mu_per_pixel, args.conf_thresh)

    outer_dict, inner_dict, center_dict = analysis_edge(dict_edges, central_name)

    inner_dict, new_inner_dict  =  counting_cutted_guv(outer_dict, inner_dict, pred_dir, args.conf_thresh, args.mu_per_pixel)

    dim_list = reconstruct_edge_map(outer_dict, new_inner_dict, center_dict, pred_dir, args.conf_thresh, args.mu_per_pixel)

    print(f"Total GUVs size list: {len(dim_list)}\n")

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
    ax.set_xlabel('GUV Diameter (μm)', fontsize=24)
    ax.set_ylabel('number of GUVs', fontsize=24)
    ax.set_ylim(0, 500)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    ax.grid(linestyle=':')
    # plt.savefig(os.path.join(folder, 'GUV_size_distribution.pdf'), dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced evaluation script for object detection models using YOLO format.")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_rgb_txt/inference",
                        help="Path to the folder containing images")
    parser.add_argument("--mu_per_pixel", type=float, default=0.3339, help="Conversion factor from pixels to micrometers")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold for predictions")
    args = parser.parse_args()

    main(args)


