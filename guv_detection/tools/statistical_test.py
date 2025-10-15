"""
Get the prediction for each acquisitions and calculate the accuracy, precision, recall, F1-score, and confusion matrix.
"""
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
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection", 
                        help="Path to the Liposomes detection folder")
    args = parser.parse_args()

    model_path = os.path.join('/home', 'angelo', 'Documenti', 'GUV_detector', f'train_{args.modality}', f'train_{args.model_size}', 'weights', 'best.pt')  # Pretrained model path
    model = ultralytics.YOLO(model_path)  # Load a custom model

    # model predint on the folder
    folder = os.path.join(args.folder, f'DATA_training_{args.modality}_txt', 'test', 'statistical_analysis')
    print(os.listdir(folder))

    for acq in os.listdir(folder):
        print(f"Processing acquisition: {acq}")
        acq_path = os.path.join(folder, acq)
        print(os.listdir(acq_path))
        results = model.predict(source=os.path.join(acq_path, 'images'), save=True, save_txt=True, save_conf=True,
                                project=acq_path)
        metrics = model.val(data=os.path.join(acq_path, 'data.yaml'),  
                            split="test",               
                            save_json=True,   
                            project=acq_path,
                            verbose=True,
                            # conf=0.25,  # Confidence threshold for predictions
                            # iou=0.95,  # IoU threshold for evaluation
                        )

        with open(os.path.join(acq_path, 'metrics.json'), 'w') as f:
            yaml.dump(metrics.results_dict, f)