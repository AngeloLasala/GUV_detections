"""
Main code for the analysis of the annotations.
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main_reader(json_path):
    """
    Main function to read the annotations
    """
    # Read the annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def annotations_analysis(annotations):
    """
    Analysis of the annotations list. It contains only he annotation of the first columns from label studio
    'original_widht': initial width of image
    'original_height': initial height of image
    'image_rotation': rotation of the image
    'value' : bbox
    'origin': type of annotation ('prediction', 'prediction-changed', 'manual')

    Parameters
    ----------
    annotations : list
        list of dict with the annotations

    Returns
    -------

    """
    # # Bounding box
    # print(annotations)
    dimensions, types, original_resolution= [], [], []
    for annotation in annotations:
        # print(annotation)
        # wh = [annotation['original_width'], annotation['original_height']]
        wh = [1024, 542]
        bbox = annotation['value']
        x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        # print(f"x = {x}, y = {y}, width = {width}, height = {height}")
        d1 = (width / 100) * wh[0]
        d2 = (height / 100) * wh[1]
        d = np.mean([d1, d2])
        # print(f'diameter: {d}, type = {annotation["origin"]}')
        original_resolution.append(wh)    
        dimensions.append(d)
        types.append('prediction')
    return dimensions, types, original_resolution

def get_analysis_dict(data):
    """
    give the json return the main analysis dictionary

    Parameters
    ----------
    data : list
        list of dta, each elements is a sigle task (image, pre-annotation, annotation)

    Returns
    -------
    data_liposomes: dict
        dictionary with the image name as key and a list with the dimensions and the type of annotation
    """
    data_liposomes = {}
    for task in data:
        name_image = task['data']['image']
        id_image = task['id']
        print(f'id {id_image}, name = {name_image}')

        ## list of dict, each for a single annotation
        annotations = task['annotations'][0]['result']    
        dimension, types, original_resolution = annotations_analysis(annotations)
        print(len(dimension))
        data_liposomes[name_image] = [dimension, types, original_resolution]
        print('=============================================')
    return data_liposomes

def main_analysis(data_liposomes, conversion_factor, json_file, parent_save_folder):
    """
    Main function to analyze the annotations

    Parameters
    ----------
    data_liposomes : dict
        dictionary with the image name as key and a list with the dimensions and the type of annotation
    """
    name_folder = json_file.split("\\")[-1].split('.')[0]
    save_folder = os.path.join(parent_save_folder, name_folder, 'analysis')
    os.makedirs(save_folder)

    total_liposomes = []
    with open(os.path.join(save_folder, 'analysis.txt'), "w") as file:
        for key in data_liposomes.keys():
            file.write(f"Image: {key}\n")
            dimensions, types, original_resolution = data_liposomes[key]
    
            dimensions = np.array(dimensions) * conversion_factor
            total_liposomes.append(dimensions)
            file.write(f"liposome detected (measure in um): {dimensions}\n")
            file.write(f"resolution: {original_resolution}\n")
            file.write(f"Number of annotations: {len(dimensions)}\n")
            if len(dimensions) > 0:
                file.write(f"Max diameter: {np.max(dimensions):.5f}\n")
                file.write(f"Min diameter: {np.min(dimensions):.5f}\n")
                file.write(f"Mean diameter: {np.mean(dimensions):.5f}\n")
                file.write(f"Median diameter: {np.median(dimensions):.5f}\n")
                file.write(f"1 quantile: {np.quantile(dimensions, 0.25):.5f}\n")
                file.write(f"3 quantile: {np.quantile(dimensions, 0.75):.5f}\n")
            else: 
                file.write('No liposomes detected\n')
            file.write('=============================================\n')

    total_liposomes = [item for sublist in total_liposomes for item in sublist]
    total_liposomes = np.array(total_liposomes)

    ## plt the istogram
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), num='histogram')
    hist = axs.hist(total_liposomes, bins=50, color='blue', alpha=0.7)
    pos = np.where(hist[0]==np.max(hist[0]))[0][0]
    peak_hist = hist[1][pos]
    axs.set_title(f'{name_folder} \n N={len(total_liposomes)} peak={peak_hist:.2f} - median={np.median(total_liposomes):.2f}, 1_qt={np.quantile(total_liposomes, 0.25):.2f}, 3_qt={np.quantile(total_liposomes, 0.75):.2f}', fontsize=18)
    axs.axvline(x = np.median(total_liposomes), c='r', linestyle='--')
    axs.set_xlabel(r'Diameter $[\mu m]$', fontsize=20)
    axs.set_ylabel('Number of Liposomes', fontsize=20)
    axs.tick_params(axis='both', labelsize=18)
    plt.savefig(os.path.join(save_folder, 'histogram.png'))

    # return '0'



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis of the annotations')
    parser.add_argument('--json_file', type=str, help='Input json file with annotation and pre-annotations, parent_directory = D:\GUV_detection\label_studio_export\#file_name.json')
    parser.add_argument('--parent_save_folder', type=str, default='D:\GUV_detection', help="Parent folder to save the analysis, default = 'D:\GUV_detection'")
    parser.add_argument('--conversion_factor', type=float, default=0.339, help='factor to convert from pixel to um, default = 0.339')
    args = parser.parse_args()

    # Read the annotations
    data = main_reader(args.json_file)
    data_liposomes = get_analysis_dict(data)
    print(args.conversion_factor)
    main_analysis(data_liposomes, args.conversion_factor, args.json_file, args.parent_save_folder)