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


def create_json(document_root, folder_path, preannotation = False):
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
        rel_path = os.path.relpath(abs_path, document_root)
        ls_path = f"/data/local-files/?d={abs_path}"
        json_data = {
            "data": {
                "image": ls_path
            },
            "annotations": [],
            "predictions": []
        }
        
        json_list.append(json_data)

        
    ## save json list as json file
    json_file_path = os.path.join(document_root, folder_path, "import.json")
    with open(json_file_path, "w") as json_file:
        json.dump(json_list, json_file, indent=4)

def main(args):

    ## create json
    create_json(args.document_root, args.folder_path, args.preannotation)

    print("✅ JSON file created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON file for preannotation")
    parser.add_argument("--document_root", type=str, default="../", help="Document root path for the images, the one that you use to set the envirorment variable")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images")
    parser.add_argument("--preannotation", action="store_true", help="Flag to indicate if load preannotation")
    args = parser.parse_args()
    
    main(args)