"""
Convert the annotation from xml (used for YOLOv3) to txt (used for YOLOv7 and YOLOv11).
"""
import time
import os
import hashlib
import argparse

from lxml import etree
import tqdm
import matplotlib.pyplot as plt

def read_xml(xml_path):
    """
    Read the xml file and return the root element, and return bbox
    """
    tree = etree.parse(xml_path)  # Replace with your actual XML file path
    root = tree.getroot()

    bboxes = []
    # Iterate through each <object> element
    for obj in root.findall('object'):
        label = obj.findtext('name')
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))

        bboxes.append({
            'label': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    return bboxes

def convert_bbox_to_yolo_format(image_path, bboxes, multi_class=False):
    """
    Convert bounding box coordinates to YOLO format.

    Parameters:
    ---------
    image_path (str): Path to the image.
    bboxes (list): List of bounding boxes, each represented as a dictionary with keys 'xmin', 'ymin', 'xmax', 'ymax'.
    multi_class (bool): If True, the function will return a list of lists, where each inner list contains the normalized coordinates for each bounding box.
    """
    ## read the image and get width and height
    img = plt.imread(image_path)
    img_height, img_width, _ = img.shape

    converted_bboxes = []
    for bbox in bboxes:
        x_center = (bbox['xmin'] + bbox['xmax']) / 2.0
        y_center = (bbox['ymin'] + bbox['ymax']) / 2.0
        width = bbox['xmax'] - bbox['xmin']
        height = bbox['ymax'] - bbox['ymin']
        if multi_class: label = bbox['label']
        else: label = 0

        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        converted_bboxes.append([label, x_center, y_center, width, height])

    return converted_bboxes

def plot_image_with_bboxes(image_path, bboxes):
    """
    Plot the image with bounding boxes.
    """
    plt.figure(figsize=(20, 10), tight_layout=True)
    img = plt.imread(image_path)
    plt.imshow(img)

    for bbox in bboxes:
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = bbox['xmax']
        ymax = bbox['ymax']

        # Create a rectangle patch
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.show()
    
def main(args):
    """
    Covert the annotation from Label Studio (in xml) to txt (used for YOLOv7 and YOLOv11).
    """
    split_list = ["train", "val", "test"]

    ## create the output folder
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # create the subfolder
    for split in split_list:
        split_path = os.path.join(args.output_path, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        if not os.path.exists(os.path.join(split_path, "images")):
            os.makedirs(os.path.join(split_path, "images"))
        if not os.path.exists(os.path.join(split_path, "labels")):
            os.makedirs(os.path.join(split_path, "labels"))

    for split in split_list:
        split_path = os.path.join(args.original_path, split)
        for img in tqdm.tqdm(os.listdir(os.path.join(split_path, "images"))):
            img_name = img.split(".")[0]
            
            img_path = os.path.join(split_path, 'images', img)
            annotation_path = os.path.join(split_path, "Annotations", img_name + ".xml")    

            # read the xml file
            bboxes = read_xml(annotation_path)
            plot_image_with_bboxes(img_path, bboxes)
            bboxes_txt = convert_bbox_to_yolo_format(img_path, bboxes, multi_class=False) 

            # create the txt file
            txt_path = os.path.join(args.output_path, split, "labels", img_name + ".txt")
            with open(txt_path, "w") as f:
                for bbox in bboxes_txt:
                    # convert to string
                    bbox_str = " ".join([str(x) for x in bbox])
                    f.write(bbox_str + "\n")
            
            # copy the image to the output folder
            output_img_path = os.path.join(args.output_path, split, "images", img)
            os.system(f"cp {img_path} {output_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the annotation from yaml (used for YOLOv3) to txt (used for YOLOv7 and YOLOv11).")
    parser.add_argument("--original_path", type=str, default='/home/angelo/Documenti/Liposomes_detection/DATA', help="Path to the original dataset.")
    parser.add_argument("--output_path", type=str, default='/home/angelo/Documenti/Liposomes_detection/DATA_txt', help="Path to the output dataset.")
    args = parser.parse_args()

    main(args)