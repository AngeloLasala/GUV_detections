"""
Train YOLO model for GUV detection
"""
import argparse
import ultralytics
import os
from ultralytics import YOLO

def parser():
    """
    define the parser for the command line arguments

    Parameters
    ----------
    None

    Returns
    -------
    args : argparse.Namespace
        model_size: str, size of YOLO model, e.g., n, s, m, l, x
        modality: str, Modality of the images: rgb or grey
        folder: str, Path to the Liposomes detection folder
        epochs: int, Number of epochs to train the model (default: 100)
    """
    parser = argparse.ArgumentParser(description="Test a model on a folder of images")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection", 
                        help="Path to the Liposomes detection folder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model")
    args = parser.parse_args()
    return args

def train(model_size, modality, folder):
    """
    Train a YOLO model for GUV detection
    """

    ## load model
    model = YOLO("yolo11n.yaml")                      # build a new model from YAML
    model = YOLO("yolo11n.pt")                        # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")   # build from YAML and transfer weights
    
    # folder
    train_folder = os.path.join(folder, f'DATA_training_{modality}_txt')

    # train the model
    results = model.train(data=os.path.join(train_folder, 'data.yaml'),  # path to data.yaml
                          epochs=100,               # number of epochs to train
                          imgsz=640,                # image size
                          project=os.path.join(train_folder, 'train'),  # project name
                         )

if __name__ == '__main__':
    args = parser()
    train(args.model_size, args.modality, args.folder)