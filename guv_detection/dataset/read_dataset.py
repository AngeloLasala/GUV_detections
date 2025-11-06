"""
Read the dataset from path. The dataset must be in the ultralytics form readi to be trained
This function provide the main statistic of the dataset
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

    boxes = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append([class_id, x_center, y_center, width, height])

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(f"{name}", fontsize=22)
        ax.imshow(image)
        ax.axis('off')

        for box in boxes:
            class_id, x_center, y_center, width, height = box
            x_center *= W
            y_center *= H
            width *= W
            height *= H

            xmin = int(x_center - (width / 2)) 
            xmax = int(x_center + (width / 2)) 
            ymin = int(y_center - (height / 2))
            ymax = int(y_center + (height / 2)) 
            
            # Create a rectangle patch
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
    return name, W, H, boxes


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

    data_split = ['test', 'train', 'val']

    dataset_info = {}
    for split in data_split:
        split_info = {}
        images_path = os.path.join(path, split, 'images')
        labels_path = os.path.join(path, split, 'labels_original')


        for image in os.listdir(images_path):
            image_path = os.path.join(images_path, image)
            label_path = os.path.join(labels_path, image.replace('.jpg', '.txt'))
            if not os.path.exists(label_path):
                continue
            name, W, H, boxes = read_image_label(image_path, label_path, show_plot=args.show_plot)
            split_info[name] = {
                'width': W,
                'height': H,
                'boxes': boxes,
                'n_boxes': len(boxes)
            }
        dataset_info[split] = split_info

    # Save dataset info to a JSON file
    output_file = os.path.join(path, 'dataset_info.json')
    with open(output_file, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    
    for split, info in dataset_info.items():
        print(f"Split: {split}")
        print(f"Number of images: {len(info)}")
        total_boxes = sum(len(data['boxes']) for data in info.values())
        print(f"Total bounding boxes: {total_boxes}")

        fig, ax = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True, num=split)
        ax[0].set_title(f"Number of Liposomes", fontsize=22)
        ax[0].hist([data['n_boxes'] for data in info.values()], bins=range(0, max(data['n_boxes'] for data in info.values()) + 2), align='left', rwidth=0.8)
        ax[0].set_xlabel('Number of Liposomes', fontsize=20)
        ax[0].set_ylabel('Frequency', fontsize=20)
        ax[0].tick_params(axis='both', which='major', labelsize=18)
        ax[0].grid(linestyle=':', color='gray')

        # plot the dimention of liposomes
        ax[1].set_title(f"Dimension of Liposomes", fontsize=22)
        widths = []
        heights = []
        for data in info.values():
            for box in data['boxes']:
                class_id, x_center, y_center, width, height = box
                widths.append(width)
                heights.append(height)
        dimention = (np.array(widths) + np.array(heights) / 2) * 640

        ax[1].hist(dimention, bins=np.arange(min(dimention), max(dimention) + 10, 10))
        ax[1].set_xlabel('Dimension (pixels)', fontsize=20)
        ax[1].set_ylabel('Frequency', fontsize=20)
        ax[1].tick_params(axis='both', which='major', labelsize=18)
        ax[1].grid(linestyle=':', color='gray')
   
    plt.show()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read dataset from path')
    parser.add_argument('--path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_rgb_txt", help='Path to the dataset')
    parser.add_argument('--show_plot', action='store_true', help="show img and bbox, default=False")
    args = parser.parse_args()

    main(args)
    