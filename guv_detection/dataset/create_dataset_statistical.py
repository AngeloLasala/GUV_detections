"""
Create the dataset for statistical analysis.
Dividing test folder in subfolder, each one contains images from a single acquisition.
"""
import os
import json
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import yaml

def main(args):
    """
    splitting test dataset
    """

    test_folder = os.path.join(args.folder, f'DATA_training_{args.modality}_txt', 'test')

    ## create the output folder
    output_folder = os.path.join(test_folder, 'statistical_analysis')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ## create the list of unique acquisition identifiers
    acquisition_ids = set()
    for img in os.listdir(os.path.join(test_folder, 'images')):
        # assuming the naming convention is something like: acquisition1_img1.png
        acquisition_id = "_".join(img.split('_')[:-1])  # Adjust this based on actual naming convention
        acquisition_ids.add(acquisition_id)
    acquisition_ids = list(acquisition_ids)

    ## create the subfolder for each acquisition
    for acq_id in acquisition_ids:
        acq_folder = os.path.join(output_folder, acq_id)
        if not os.path.exists(acq_folder):
            os.makedirs(acq_folder)
        if not os.path.exists(os.path.join(acq_folder, 'images')):
            os.makedirs(os.path.join(acq_folder, 'images'))
        if not os.path.exists(os.path.join(acq_folder, 'labels')):
            os.makedirs(os.path.join(acq_folder, 'labels'))

    # move the images and labels to the respective subfolder
    for img in os.listdir(os.path.join(test_folder, 'images')):
        acquisition_id = "_".join(img.split('_')[:-1])
        src_img_path = os.path.join(test_folder, 'images', img)
        src_label_path = os.path.join(test_folder, 'labels', img.replace('.jpg', '.txt').replace('.png', '.txt'))
        dst_img_path = os.path.join(output_folder, acquisition_id, 'images', img)
        dst_label_path = os.path.join(output_folder, acquisition_id, 'labels', img.replace('.jpg', '.txt').replace('.png', '.txt'))
        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_label_path, dst_label_path)
    print(f"Dataset for statistical analysis created at {output_folder}")

    ## create yaml file for each acquisition
    for acq_id in acquisition_ids:
        acq_folder = os.path.join(output_folder, acq_id)
        num_classes = 1
        names = ['GUV']
        yaml_content = {
            'test': os.path.join(acq_folder, 'images'),
            'nc': num_classes,
            'names': names 
        }
        yaml_path = os.path.join(acq_folder, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=True, sort_keys=False)
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the dataset for statistical analysis.")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection", 
                        help="Path to the Liposomes detection folder")
    args = parser.parse_args()

    main(args)