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

## GUV Detector
To do ...

## Label Studio Inteface - Create GUV project

Label studio is an open source platform usefull for creating a userfrandly interface to laod an annotate dataset for diverse type of ML project. Here, we provide a simple guideline for creating a project releated to GUV detection

For the sake of simplicity, in this project, we guide the  installation process using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

### Create the GUV Detection project
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

Here an example of label studio project for GUV manual annotation
![example of label-studio project](guv_detection/images/example_ls.png)

Once the annotations are done, go to **Export** -> **YOLO with Images** to get images and label ready to be use for training and/or testing Yolov11 model!!

### Import Images and Pre-annotation with JSON file

A uselfull way to import data in the label studio project is via JSON file. Here, a guideline to import your data, and eventually pre-annotations/annotations, by building a single JSON file. 
For details about the JSON format and the preannotation see [Basic Label studio JSON format](https://labelstud.io/guide/tasks#Basic-Label-Studio-JSON-format) and [Import pre-annotated data into Label Studio](https://labelstud.io/guide/predictions)

Before open the label-studio project, you have to lunch wiht the permission to access a local path. the configuration of local file must be as follow:

```
| document_root
|   |-- folder_path
|   |   |-- image_1.jpg
|   |   |-- image_2.jpg
|   |   |-- image_n.jpg
```

Set the correct envirorment configuration to have axcess to the local file

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=document_root
```

Then lunch label-studio:

```bash
label-studio
```

Open the project (see *Label Studio Interface - Create GUV procet* for details), and follow this guideline for setting the local file envirormnet for [Settupping connection in Label Studio UI](https://labelstud.io/guide/storage#Set-up-connection-in-the-Label-Studio-UI-4)

- Open **Setting > Cloude Storage**
- Click **Add Source Storage**
- Select **Local Files** as the storage type
- Insert a name for your storage title: example Leica Angelo
- Specify an **Absolute local path** to the directory with your files. For the tree configuration above insert  `document_root/folder_path`. Then **Verified connect** and click **Next**
- *(Optional)* In the File Filter Regex field, specify a regular expression to filter bucket objects. Use .* to collect all objects.
- *(Optional)* In the Import method dropdown, choose how to import your data:
    - Files - Automatically creates a task for each storage object (e.g. JPG, MP3, TXT). Use this if you want to create Label Studio tasks from media files automatically. Use this option for labeling configurations with one source tag.
    - Tasks - Treat each JSON, JSONL, or Parquet as a task definition (one or more tasks per file). Use this if you want to import tasks in Label Studio JSON format directly from your storage. Use this option for complex labeling configurations with HyperText or multiple source tags.
- Click **Next**
- Finally click **Save**. **DO NOT** save and sync

Here an example of storage information
![example of label-studio storage informatio](guv_detection/images/local_file_storage.png)
Again, **do not click on sync** in this phase.

- Go to the **Project > Import**
- Inport a file  `import.json`






## Usage for custom training

### Dataset
The dataset used to train and validate the GUV Detector consists of microscopy images of GUVs, annotated by expert researchers in the field. The data is organized as follows:

```
| DATA
|   |-- train
|   |   |-- images
|   |   |   |-- image_1.jpg
|   |   |   |-- image_2.jpg
|   |   |   |-- ...
|   |   |-- labels
|   |   |   |-- image_1.txt
|   |   |   |-- image_2.txt
|   |   |   |-- ...
|   |-- val
|   |   |-- images
|   |   |-- labels
|   |-- test
|   |   |-- images
|   |   |-- labels
```

The current version of the project supports both .png and .jpg image formats.
Each corresponding label.txt file must follow the Ultralytics YOLO annotation format. Please refer to the official Ultralytics documentation for detailed [annotation guidelines](https://docs.ultralytics.com/it/datasets/detect/#supported-dataset-formats) .

To ensure compatibility and prevent errors, use the **Label Studio Interface – Create GUV Project** guide above to take full advantage of Label Studio for annotation!

### Training

Once the dataset is prepared, you can train your custom model using the `train.py` script.

```bash
python train.py --model --epoch --folder --modality
```

The dataset folder used for training must be organized as follows:
```
| folder
|   |-- modality_1
|   |   |-- DATA
|   |-- modality_2
|   |   |-- DATA
```
Each modality (e.g., `rgb`, `grey`) should contain its own DATA directory with the standard `train`, `val`, and `test` subfolders.

### Test
To do...

