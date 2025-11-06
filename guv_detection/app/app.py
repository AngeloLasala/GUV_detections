"""
GUV_app with simple GUI for using GUV detector for real experiment
"""
import os
import argparse
import ultralytics
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import numpy as np

def read_pred_boxes(pred_path, conf_thresh):
    """Read predicted boxes with confidence filtering."""
    boxes = []
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls, xc, yc, w, h, conf = map(float, parts[:6])
                    if conf >= conf_thresh:
                        boxes.append([xc, yc, w, h, conf])
    return boxes

def processing_input_folder(folder, split_factor):
    """
    Processing imput folder to cropping the inner images fro detecting liposomes.
    Note that the images must have the same resolution for a correct identification of best cropping

    Parameters
    ----------
    folder: str
            input folder
    split_factor: int, 2 or 4
                2 -> split img in 4 sub imgs
                4 -> split img in 16 sub imgs

    Returns
    -------
    sub_folder: str
            sub folder of cropping images
    """

    if split_factor not in [2, 4]:
        raise ValueError("❌ split_factor must be 2 o 4.")
    
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError("❌ Not images found!!")


    # check the resolution of fosrt image
    first_image_path = os.path.join(folder, image_files[0])
    try:
        with Image.open(first_image_path) as img:
            reference_size = img.size  # (W, H)
    except Exception as e:
        raise ValueError(f"Not image found: {first_image_path}\nError: {e}")


    # select imges with resolution equal to the reference one
    inconsistent_images = []
    processing_images = [first_image_path]
    for img_name in image_files[1:]:
        img_path = os.path.join(folder, img_name)
        if img.size != reference_size:
            inconsistent_images.append(img_path)
        else:
            processing_images.append(img_path)

    ## create processing subfolder 
    sub_folder = os.path.join(folder, 'processing_images')
    os.makedirs(sub_folder, exist_ok=True)
    

    for img_path in processing_images:
        img = Image.open(img_path)

        w, h = img.size
        step_w = w // split_factor
        step_h = h // split_factor

        crop_boxes = []
        for i in range(split_factor):
            for j in range(split_factor):
                left = j * step_w
                upper = i * step_h
                right = left + step_w
                lower = upper + step_h
                crop_boxes.append((left, upper, right, lower))

        for i, box in enumerate(crop_boxes):
            cropped_img = img.crop(box)
            cropped_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_crop{i+1}.png"
            cropped_path = os.path.join(sub_folder, cropped_name)
            cropped_img.save(cropped_path)

    print('Processing input folder done!!')
    return sub_folder


def main(args):
    """
    Main function. Performe the analysis across the folder of images
    """

    ## read the model
    model_path = os.path.join('model', args.modality, f'yolo11_{args.model_size}', 'best.pt')
    model = ultralytics.YOLO(model_path)

    ## processing input folder 
    sub_folder = processing_input_folder(args.folder, args.split_factor)

    ## model prediction on the folder
    results = model.predict(source=sub_folder, save=True, save_txt=True, save_conf=True,
                            project=sub_folder)

    # check if the predict formde is create
    if not os.path.exists(os.path.join(sub_folder, 'predict')):
        print("No predictions found. Please check the model and input folder.")
        return None
    
    prediction_folder = os.path.join(sub_folder, 'predict', 'labels')

    count_edge = 0
    count_inter = 0
    conf_thresh = 0.25
    dim_list = []
    for i in os.listdir(prediction_folder):
        if i.endswith('.txt'):
            pred_path = os.path.join(prediction_folder, i)
            image_name = os.path.join(sub_folder, 'predict', i.replace('.txt', '.jpg'))
            w, h = Image.open(image_name).size
            boxes = read_pred_boxes(pred_path, conf_thresh)
            for bbox in boxes:
                w_guv = bbox[2] * w 
                h_guv = bbox[3] * h

                max_dim = max(w_guv, h_guv)
                min_dim = min(w_guv, h_guv)

                if min_dim <= 0.5 * max_dim:
                    count_edge += 1
                else:
                    count_inter += 1
                
                dim = max_dim * args.mu_per_pixel
                dim_list.append(dim)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), tight_layout=True)
    bin_width = 5
    bins = np.arange(0, max(dim_list) + bin_width, bin_width)
    ax.hist(dim_list, bins=bins, color='chocolate', alpha=0.5, density=True)
    median = np.median(dim_list)
    first_quartile = np.percentile(dim_list, 25)
    third_quartile = np.percentile(dim_list, 75)
    ax.axvline(median, color='darkred', linestyle='dashed', linewidth=3, label=f'Median: {median:.2f} μm')
    ax.axvline(first_quartile, color='red', linestyle='dashed', linewidth=3, label=f'Q1: {first_quartile:.2f} μm')
    ax.axvline(third_quartile, color='red', linestyle='dashed', linewidth=3, label=f'Q3: {third_quartile:.2f} μm')

    # lognormal fit
    shape, loc, scale = lognorm.fit(dim_list, floc=0)  # floc=0 forza il fit standard lognormale
    x = np.linspace(0, max(dim_list), 1000)
    n_guv = len(dim_list)
    mu = np.log(scale)
    sigma = shape
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
    ax.plot(x, pdf, 'k-', linewidth=3, label=(
        f'Log-normal fit\nμ={mu:.2f}, σ={sigma:.2f}\n'
        f'Total GUVs: {n_guv}'
    ))

    # set x label
    ax.set_xlabel('GUV Diameter (μm)', fontsize=24)
    ax.set_ylabel('density of GUVs', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    ax.grid(linestyle=':')
    plt.savefig(os.path.join(sub_folder, 'GUV_size_distribution.pdf'), dpi=300)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone app for automatic liposomes detection")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--modality", type=str, default="rgb", help="Modality of the images: rgb or grey")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection/DATA_training_txt/inference", 
                        help="Path to the folder containing images")
    parser.add_argument("--split_factor", type=int, default=2, help="Splitting faction for input images, 2 or 4")
    parser.add_argument("--mu_per_pixel", type=float, default=0.3339, help="Conversion factor from pixels to micrometers")
    args = parser.parse_args()

    main(args)