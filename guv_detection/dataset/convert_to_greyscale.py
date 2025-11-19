"""
Create an equivalent dataset converting the images in greyscale
"""
import os
import json
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def read_image_label(image_path, label_path, show_plot=False):
    """
    Function to plot the image and label from given path

    Parameters:
    ----------
    image_path : str
        Path to the image file
    label_path : str
        Path to the label file
    """
    name = os.path.basename(image_path.split('.')[0])
    image = Image.open(image_path)
    W, H = image.size

    # read the txt file line by line
    parent_labels = os.path.dirname(os.path.dirname(label_path))
    new_label_path = os.path.join(parent_labels, 'labels')
    

def main(args):
    """
    Main function to read the dataset and print statistics

    Parameters:
    ----------
    args : argparse.Namespace
        Command line arguments containing the dataset path

    Returns:
    -------
    """
    path = args.path
    path_grey = args.path_grey
    data_split = args.folders

    # Check the requested splits exist in the source path; skip those that don't
    available_splits = []
    for split in data_split:
        split_images = os.path.join(path, split, 'images')
        split_labels = os.path.join(path, split, 'labels')
        if not os.path.exists(split_images):
            print(f"[WARN] Source images folder not found for split '{split}': {split_images} -> skipping this split.")
            continue
        if not os.path.exists(split_labels):
            print(f"[WARN] Source labels folder not found for split '{split}': {split_labels} -> skipping this split.")
            continue
        available_splits.append(split)

    if not available_splits:
        raise FileNotFoundError(f"No requested splits found under {path}. Requested splits: {data_split}")

    # create grey dataset folder structure only for the available splits
    for split in available_splits:
        images_out = os.path.join(path_grey, split, 'images')
        labels_out = os.path.join(path_grey, split, 'labels')
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)


    dataset_info = {}
    for split in available_splits:
        print('Processing split:', split)
        split_info = {}
        images_path = os.path.join(path, split, 'images')
        labels_path = os.path.join(path, split, 'labels')

        for image in tqdm(os.listdir(images_path)):
            # chek if image end with .jpd or .png
            if image.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_path, image)
            else:
                pass
            label_path = os.path.join(labels_path, image.replace('.jpg', '.txt'))

            image_gray = Image.open(image_path).convert('L')
            image_gray.save(os.path.join(path_grey, split, 'images', image))

            # copy the label file to the new path
            new_label_path = os.path.join(path_grey, split, 'labels', image.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                with open(new_label_path, 'w') as f:
                    f.writelines(lines)
            else:
                print(f"Label file {label_path} not found. Skipping.")
                continue
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read dataset from path')
    parser.add_argument('--path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_rgb_txt", help='Path to the dataset')
    parser.add_argument('--path_grey', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_grey_txt", help='Path to the gray dataset')
    parser.add_argument('--folders', type=str, nargs='+', default=['train', 'val', 'test'], help='Folders to process')

    args = parser.parse_args()

    main(args)