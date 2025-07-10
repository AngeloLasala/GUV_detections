"""
Read the dataset from path. The dataset must be in the ultralytics form readi to be trained
This function provide the main statistic of the dataset
"""
import os
import json
import argparse
import numpy as np 
import matplotlib.pyplot as plt

def plot_image_label(image_path, label_path):
    """
    Function to plot the image and label from given path

    Parameters:
    ----------
    image_path : str
        Path to the image file
    label_path : str
        Path to the label file
    """
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

    ##  check if 'train' 'val' 'test' folders are present
    if not os.path.exists(os.path.join(path, 'train')):
        raise FileNotFoundError(f"Train folder not found in {path}")
    if not os.path.exists(os.path.join(path, 'val')):
        raise FileNotFoundError(f"Validation folder not found in {path}")
    if not os.path.exists(os.path.join(path, 'test')):
        raise FileNotFoundError(f"Test folder not found in {path}")

    data_split = ['train', 'val', 'test']

    for split in data_split:
        images_path = os.path.join(path, split, 'images')
        labels_path = os.path.join(path, split, 'labels')

        # Check if images and labels folders exist
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Images folder not found in {images_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels folder not found in {labels_path}")
        
        for i,j in zip(os.listdir(images_path), os.listdir(labels_path)):
            print(f"Image: {i}, Label: {j}")

        # # Initialize counters
        # total_images = 0
        # total_boxes = 0

        # for txt_file in txt_files:
        #     with open(os.path.join(split_path, txt_file), 'r') as file:
        #         lines = file.readlines()
        #         total_images += 1
        #         total_boxes += len(lines)

        # print(f"Total images in {split}: {total_images}")
        # print(f"Total boxes in {split}: {total_boxes}")
        # print(f"Average boxes per image in {split}: {total_boxes / total_images if total_images > 0 else 0:.2f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read dataset from path')
    parser.add_argument('--path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_txt", help='Path to the dataset')
    args = parser.parse_args()

    main(args)
    