"""
Convert the annotation from xml (used for YOLOv3) to txt (used for YOLOv7 and YOLOv11).
"""
import time
import os
import hashlib
import argparse
import cv2
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
    

    # Calculate the dimensions of the cropped images
    crop_width = img_width // 2
    crop_height = img_height // 2



    converted_bboxes_tl = []
    converted_bboxes_tr = []
    converted_bboxes_bl = []
    converted_bboxes_br = []
    for bbox in bboxes:
        x_center = (bbox['xmin'] + bbox['xmax']) / 2.0
        y_center = (bbox['ymin'] + bbox['ymax']) / 2.0
        width = bbox['xmax'] - bbox['xmin']
        height = bbox['ymax'] - bbox['ymin']
        if multi_class: label = bbox['label']
        else: label = 0

        #x_center /= img_width
        #y_center /= img_height
        width /= img_width
        height /= img_height

        if (x_center <= crop_width) & (y_center>= crop_height) :
            converted_bboxes_bl.append([label, x_center/img_width, (y_center-crop_height)/img_height, width, height])
        elif (x_center >= crop_width) & (y_center>= crop_height):
            converted_bboxes_br.append([label, (x_center-crop_width)/img_width, (y_center-crop_height)/img_height, width, height])
        elif (x_center <= crop_width) & (y_center<= crop_height):
            converted_bboxes_tl.append([label, x_center/img_width, y_center/img_height, width, height])
        elif (x_center >= crop_width) & (y_center <= crop_height):
            converted_bboxes_tr.append([label, (x_center-crop_width)/img_width, y_center/img_height, width, height])

    return converted_bboxes_tl,converted_bboxes_tr,converted_bboxes_bl,converted_bboxes_br
    

def plot_image_with_bboxes(image_path, bboxes,img_width,img_height): #no needed 
    """
    Plot the image with bounding boxes.
    """
    plt.figure(figsize=(20, 10), tight_layout=True)
    img = plt.imread(image_path)
    plt.imshow(img)
    # to do: convert coordinates of bboxes to report them into the cropped images  
    for bbox in bboxes:
    
        xcnt = bbox[1]*img_width
        ycnt = bbox[2]*img_height
        

        # Create a rectangle patch
        rect = plt.Rectangle((xcnt-(bbox[3]/2*img_width), ycnt-(bbox[4]/2*img_height)), bbox[3]*img_width, bbox[4]*img_height, linewidth=1, edgecolor='r', facecolor='none')
        cnt = plt.Circle((xcnt, ycnt),2, color='red')
        plt.gca().add_patch(rect)
        plt.gca().add_patch(cnt)
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
        if not os.path.exists(os.path.join(split_path, "labels")):# ccartella dove vanno i .txt files 
            os.makedirs(os.path.join(split_path, "labels"))

    for split in split_list:
        split_path_i = os.path.join(args.original_path, split) # ok original path

        for img in tqdm.tqdm(os.listdir(os.path.join(split_path_i, "images"))): # split_path_i= original img_path 
            img_name = img.split(".")[0]
            
            img_path = os.path.join(split_path_i, 'images', img)
            annotation_path = os.path.join(split_path_i, "Annotations", img_name + ".xml") 
            ###############################################################################################################
            # put the cropped images in the 'images' folder of the output folder
            

            image = cv2.imread(img_path)
            
            # Get the dimensions of the image
            height, width, _ = image.shape

            # Calculate the dimensions of the cropped images
            crop_width = width // 2
            crop_height = height // 2

            # Crop the image into four parts
            top_left = image[:crop_height, :crop_width]
            top_right = image[:crop_height, crop_width:]
            bottom_left = image[crop_height:, :crop_width]
            bottom_right = image[crop_height:, crop_width:]
            

            # save images in the new folder
            cv2.imwrite(os.path.join(os.path.join(args.output_path, split, "images"), f'{img_name}_tl.jpg'), top_left)
            cv2.imwrite(os.path.join(os.path.join(args.output_path, split, "images"), f'{img_name}_tr.jpg'), top_right)
            cv2.imwrite(os.path.join(os.path.join(args.output_path, split, "images"), f'{img_name}_bl.jpg'), bottom_left)
            cv2.imwrite(os.path.join(os.path.join(args.output_path, split, "images"), f'{img_name}_br.jpg'), bottom_right)




            ##################################################################################################################

            # read the xml file
            bboxes = read_xml(annotation_path)#annotation path of the original images folder
            #plot_image_with_bboxes(img_path, bboxes, img_width=args.original_image_width,img_height=args.original_image_height) # img_path = path of the images original images
            
            #save the coordinates of bounding boxes center to 4 different list to be written in 4 different .txt files 
            bboxes_txt_tl, bboxes_txt_tr, bboxes_txt_bl, bboxes_txt_br = convert_bbox_to_yolo_format(img_path, bboxes, multi_class=False)
            bboxes_txt_list=[bboxes_txt_tl, bboxes_txt_tr, bboxes_txt_bl, bboxes_txt_br]
            
           
            
            

            bboxes_txt_dict={'0':"_tl",'1':"_tr",'2':"_bl",'3':"_br"}
            # create the txt file
            for idx,bboxes_txt in enumerate(bboxes_txt_list):
                txt_path = os.path.join(args.output_path, split, "labels", img_name + bboxes_txt_dict[str(idx)] + ".txt")
                plot_image_with_bboxes(os.path.join(args.output_path, split, "images", img_name + bboxes_txt_dict[str(idx)] + ".jpg"), bboxes_txt,width,height)
                
                with open(txt_path, "w") as f:
                    for bbox in bboxes_txt:
                    # convert to string
                        bbox_str = " ".join([str(x) for x in bbox])
                        f.write(bbox_str + "\n")
            
            # copy the image to the output folder
                #output_img_path = os.path.join(args.output_path, split, "images", img)
                #os.system(f"cp {img_path} {output_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the annotation from yaml (used for YOLOv3) to txt (used for YOLOv7 and YOLOv11).")
    parser.add_argument("--original_path", type=str, default='/home/angelo/Documenti/Liposomes_detection/DATA', help="Path to the original dataset.") # original DATA path with cropped images 
    parser.add_argument("--output_path", type=str, default='/home/angelo/Documenti/Liposomes_detection/DATA_txt', help="Path to the output dataset.")
    args = parser.parse_args()

    main(args)

    