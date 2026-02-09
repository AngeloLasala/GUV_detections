"""
Adding addtional labelled data for improving the performance od GUV detection.
The main scructture of the code is the following:
- Main dataset: path to the main dataset, e.g. "DATA_training_rgb_tx" (with train val)
- Additional dataset: path to the additional dataset, e.g. "Data-Dario-Dicembre" (single folder named test)

From the additional datset, random samplin 80-20 is applied to extrapolate train data and val data 
and added to the main dataset. Resulting dataset is save as follows:
- Final dataset: in the same folder of main dataset (e.g. "DATA_training_rgb_txt") create
                 train_adding_#Additional_dataset 
                 val_addind_#Additional_dataset
"""
import os
import argparse
import shutil
import numpy as np

def main(args):
    """
    """
    main_dataset_path = args.path_main
    additional_dataset_path = args.path_additional


    # Create new folders for the final dataset
    train_out = os.path.join(args.path_main, f'train_adding_{os.path.basename(args.path_additional)}')
    val_out = os.path.join(args.path_main, f'val_adding_{os.path.basename(args.path_additional)}')
    os.makedirs(os.path.join(train_out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_out, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_out, 'labels'), exist_ok=True)

    # Copy original main dataset images and labels to the new folders
    for split in ['train', 'val']:
        images_path = os.path.join(args.path_main, split, 'images')
        labels_path = os.path.join(args.path_main, split, 'labels')

        for image in os.listdir(images_path):
            if image.lower().endswith(('.jpg')):
                if split == 'train':
                    shutil.copy(os.path.join(images_path, image), os.path.join(train_out, 'images', image))
                    label_file = image.replace('.jpg', '.txt')
                    shutil.copy(os.path.join(labels_path, label_file), os.path.join(train_out, 'labels', label_file))
                elif split == 'val':
                    shutil.copy(os.path.join(images_path, image), os.path.join(val_out, 'images', image))
                    label_file = image.replace('.jpg', '.txt')
                    shutil.copy(os.path.join(labels_path, label_file), os.path.join(val_out, 'labels', label_file))
               
    # # Randomly sample 80% of the additional dataset for training and 20% for validation
    # set seed for reproducibility
    sedd = 42
    np.random.seed(sedd)
    
    additional_images_path = os.path.join(args.path_additional, 'test', 'images')
    additional_labels_path = os.path.join(args.path_additional, 'test', 'labels')
    additional_images = [f for f in os.listdir(additional_images_path) if f.lower().endswith(('.jpg'))]
    np.random.shuffle(additional_images)
    split_index = int(0.8 * len(additional_images))
    train_additional_images = additional_images[:split_index]
    val_additional_images = additional_images[split_index:]

    # Copy the additional images and labels to the new folders
    for image in train_additional_images:
        shutil.copy(os.path.join(additional_images_path, image), os.path.join(train_out, 'images', image))
        label_file = image.replace('.jpg', '.txt')
        shutil.copy(os.path.join(additional_labels_path, label_file), os.path.join(train_out, 'labels', label_file))
    
    for image in val_additional_images:
        shutil.copy(os.path.join(additional_images_path, image), os.path.join(val_out, 'images', image))
        label_file = image.replace('.jpg', '.txt')
        shutil.copy(os.path.join(additional_labels_path, label_file), os.path.join(val_out, 'labels', label_file))
    
    print(f"Added {len(train_additional_images)} images to the training set and {len(val_additional_images)} images to the validation set from the additional dataset '{os.path.basename(args.path_additional)}'.")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read dataset from path')
    parser.add_argument('--path_main', type=str, default="/media/angelo/PortableSSD/PhD_notes/Liposome detection/DATA_training_rgb_txt", help='Path to the main dataset')
    parser.add_argument('--path_additional', type=str, default="/media/angelo/PortableSSD/PhD_notes/Liposome detection/DATA_training_rgb_txt/Leica-Dicembre-Dario", help='Path to the additional dataset')

    args = parser.parse_args()

    main(args)
    