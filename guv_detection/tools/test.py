"""
Testing the model on a folder with microscop image of GUV 
"""
import os
import argparse
import ultralytics
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model on a folder of images")
    parser.add_argument("--model", type=str, default="/home/angelo/Documenti/GUV_detector",
                                             help="Path to the model file")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_txt", 
                        help="Path to the Liposomes detection folder")
    args = parser.parse_args()

    model_path = os.path.join(args.model, f'train_{args.modality}', 'weights', 'best.pt')  
    model = ultralytics.YOLO(model_path)

    # model predint on the folder
    folder = os.path.join(args.folder, f'DATA_training_{args.modality}_txt')
    test_path = os.path.join(folder, 'test', 'images')
    results = model.predict(source=os.path.join(folder, 'test', 'images'), save=True, save_txt=True, save_conf=True,
                            project=os.path.join(folder, 'test'))

    # model val on the test set
    metrics = model.val(data=os.path.join(folder, 'data.yaml'),  
                        split="test",               
                        save_json=True,   
                        project=os.path.join(folder, 'test'),
                        verbose=True,
                        # conf=0.25,  # Confidence threshold for predictions
                        # iou=0.95,  # IoU threshold for evaluation
                    )
                        



  
