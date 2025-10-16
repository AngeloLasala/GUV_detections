"""
Create an equivalent dataset converting the images in greyscale
"""
import os
import json
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

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
    path_gray = args.path_gray

    ##  check if 'train' 'val' 'test' folders are present
    if not os.path.exists(os.path.join(path, 'train')):
        raise FileNotFoundError(f"Train folder not found in {path}")
    if not os.path.exists(os.path.join(path, 'val')):
        raise FileNotFoundError(f"Validation folder not found in {path}")
    if not os.path.exists(os.path.join(path, 'test')):
        raise FileNotFoundError(f"Test folder not found in {path}")

    # check if the gray path exist, if not create it
    if not os.path.exists(path_gray):
        os.makedirs(path_gray)
        os.makedirs(os.path.join(path_gray, 'train', 'images'))
        os.makedirs(os.path.join(path_gray, 'train', 'labels'))
        os.makedirs(os.path.join(path_gray, 'val', 'images'))
        os.makedirs(os.path.join(path_gray, 'val', 'labels'))
        os.makedirs(os.path.join(path_gray, 'test', 'images'))
        os.makedirs(os.path.join(path_gray, 'test', 'labels'))
    else: 
        print(f"Gray path {path_gray} already exists. Please remove it or choose another path.")
        return None


    data_split = args.folders

    dataset_info = {}
    for split in data_split:
        print('Processing split:', split)
        split_info = {}
        images_path = os.path.join(path, split, 'images')
        labels_path = os.path.join(path, split, 'labels')

        for image in os.listdir(images_path):
            image_path = os.path.join(images_path, image)
            label_path = os.path.join(labels_path, image.replace('.jpg', '.txt'))

            image_gray = Image.open(image_path).convert('L')
            image_gray.save(os.path.join(path_gray, split, 'images', image))

            # copy the label file to the new path
            new_label_path = os.path.join(path_gray, split, 'labels', image.replace('.jpg', '.txt'))
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
    parser.add_argument('--path_gray', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_grey_txt", help='Path to the gray dataset')
    parser.add_argument('--folders', type=str, nargs='+', default=['train', 'val', 'test'], help='Folders to process')


    args = parser.parse_args()

    main(args)