# GUV Detections
Automatic detection of **Giant Unilamellar Vesicle (GUV)** in miscroscopic images using YOLOv11

![example of detection](guv_detection/images/GR06_20240326_NIK_P17_E018_01_05_B1.jpg)

This project implements an object detection pipeline using the YOLOv11 algorithm. The system is trained on a custom dataset manually annotated by expert researchers for the localization of GUVs in microscopy images. Although the project is specifically designed for GUV detection, the pipeline is fully generalizable and can be adapted to other types of images.

Check out the [YOLOv11 by Ultralytics](https://docs.ultralytics.com/it/models/yolo11/) for more detailed information about object detection model.

## Install
Installation guideline is based on the Anaconda/Miniconda environment. To install Miniconda, refer to the [official documentation](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).


Create virtual envirorment with ppython version 3.10:

```bash
conda create --name guv python=3.10
```

Deactivate and activate it with the following comands:

```bash
conda deactivate
conda activate guv
``` 

Clone the repository to your local machine. If git in not installed in new env use `conda install git`.
```bash
git clone git@github.com:AngeloLasala/GUV_detections.git
```

Move on `\guv_detection` girectory and install packeges with the following comand:
```bash
pip install -e .
```

**!! Note !!**: The Ultralytics package automatically installs the necessary NVIDIA and CUDA dependencies required for GPU usage.

## Label Studio Inteface - create a project

The inference phase is the stage during which the trained model is utilized for real-world applications. The detailed procedure to install Label-studio in a virtual envirorment is [here](https://labelstud.io/guide/install.html). 

For the sake of simplicity, in this project, we guide the  installation process using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) and the [ML backend interface](https://labelstud.io/guide/ml.html) .

## Create the Liposomes_Detection project
First of all create a virtul env where install the requirement packeges for Label-studio. For Anacondo/Miniconda env:
```bash
conda create --name label-studio python=3.10
conda activate label-studio
conda install psycopg2
pip install label-studio
```

Open the label-studio interface with the following comand lines:
```bash
conda activate label-studio
label-studio
```

Create an object detection progect following the label-studio interface. The code view for pure object detection (*exp_1*) is:
```
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    
    
  <Label value="Empty_GUV" background="#FFA39E"/></RectangleLabels>
</View>
```
At this stage, you can load the images and start the manual annotations.

**NOTE!!**: the Label studio interface runs on a localholst 8080 that must be activate during the usage. The suggestion is to open an *Anaconda prompt (Miniconda)* and lanch the above code without closing the windows.

