"""
Testing the model on a folder with microscop image of GUV 
"""
import os
import argparse
import ultralytics
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model on a folder of images")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--main_folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection", 
                        help="Path to the Liposomes detection folder")
    parser.add_argument("--folder", type=str, default="test_folder", help="test folder, it has 'test' subgolder with 'images' and 'labels'")
    parser.add_argument("--model_folder", type=str, default=os.path.join('C:\\', 'Users', 'lasal', 'Documents', 'GUV_detector'),
                        help="Path to the GUV_detector folder containing trained models")

    args = parser.parse_args()

    model_path = os.path.join(args.model_folder, f'train_{args.modality}', f'train_{args.model_size}', 'weights', 'best.pt')  # Pretrained model path
    model = ultralytics.YOLO(model_path)  # Load a custom model

    # model predint on the folder
    folder = os.path.join(args.main_folder, f'DATA_training_{args.modality}_txt')
    project_path = os.path.join(folder, args.folder)
    results = model.predict(source=os.path.join(project_path, 'test', 'images'), save=True, save_txt=True, save_conf=True,
                            project=project_path)

    # model val on the test set
    metrics = model.val(data=os.path.join(project_path, "data.yaml"),  
                        split="test",               
                        save_json=True,   
                        project=project_path,
                        verbose=True,
                        # conf=0.25,  # Confidence threshold for predictions
                        # iou=0.95,  # IoU threshold for evaluation
                    )



  
