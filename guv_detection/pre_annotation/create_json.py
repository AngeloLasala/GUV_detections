"""
Create JSON file to import data with preannotation. for a single 'task' the simple format of json file is as follows:

{
  "id": 151,
  "data": {
    "image": "/data/upload/4/84757ee9-20240515_LEI_P17_E025_FP.lif_-_Image001_crop1.png"
  },
  "annotations": [],
  "predictions": []
}

"""

import os
import json
import argparse
from urllib.parse import quote
from guv_detection.tools.inference import read_pred_boxes
from PIL import Image


def create_json(document_root, folder_path, conf_thresh):
    """
    Create json file for each images in the folder_path
    """
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    images = [f for f in os.listdir(os.path.join(document_root, folder_path)) if f.lower().endswith(valid_ext)]
    
    if not images:
        print("Not images found in the folder.")
        return
    
    json_list = []
    for img_name in images:   ## this is the for loop for task, i.e. for each images
        abs_path = os.path.join(folder_path, img_name)
        ls_path = f"/data/local-files/?d={abs_path}"
        
        json_data = {
            "data": {
                "image": ls_path
            },
            "annotations": [],
            "predictions": []
        }

        ## ADD PREANNOTATION
        total_path = os.path.join(document_root, folder_path)
        labels = os.path.join(total_path, "predict", "labels")

        image_path = os.path.join(document_root, folder_path, img_name)
        W, H = Image.open(image_path).size

        if os.path.exists(labels):
            result = []
            img_name = os.path.splitext(img_name)[0]
            preannotation_path = os.path.join(labels, img_name + ".txt")
            boxes = read_pred_boxes(preannotation_path, conf_thresh)

            for bbox in boxes:
                xc, yc, w, h, c = bbox
                x1 = xc - w / 2
                y1 = yc - h / 2
                
                result.append({
                    "original_width": W,
                    "original_height": H,
                    "image_rotation": 0,
                    "value": {
                        "x": x1 * 100,
                        "y": y1 * 100,
                        "width": w * 100,
                        "height": h * 100,
                        'rotation': 0, 
                        'rectanglelabels': ['Empty_GUV']
                    },
                    "from_name": "label",
                    "to_name": "image",
                    'type': 'rectanglelabels', 
                    'origin': 'manual'
                })
            
            prediction = []
            prediction.append({
                'result': result})

            json_data["predictions"] = prediction
        else:
            print(f"No preannotation found for {img_name}")
        
        json_list.append(json_data)

        
    ## save json list as json file
    json_file_path = os.path.join(document_root, folder_path, "import.json")
    with open(json_file_path, "w") as json_file:
        json.dump(json_list, json_file, indent=4)

def main(args):

    ## create json
    create_json(args.document_root, args.folder_path, args.conf_thresh)

    print("✅ JSON file created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON file for preannotation")
    parser.add_argument("--document_root", type=str, default="../", help="Document root path for the images, the one that you use to set the envirorment variable")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images")
    parser.add_argument("--preannotation", action="store_true", help="Flag to indicate if load preannotation")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold for predictions")

    args = parser.parse_args()
    
    main(args)