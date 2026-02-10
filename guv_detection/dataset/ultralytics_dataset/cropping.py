"""
Cropping imges in folder in 4 or 16 subimges
"""
import os
import argparse
from PIL import Image

def crop_images(input_folder, num_crops):
    """
    crop images in a folder into 4 or 16 subimages and save them in a new folder
    """

    if num_crops not in (4, 16):
        raise ValueError("Il numero di crop deve essere 4 o 16")

    split_factor = int(num_crops ** 0.5)

    output_folder = os.path.join(input_folder, f"cropped_{num_crops}")
    os.makedirs(output_folder, exist_ok=True)

    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

    if not images:
        print("Nessuna immagine trovata nella cartella.")
        return

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path)

        w, h = img.size
        step_w = w // split_factor
        step_h = h // split_factor

        crop_idx = 1
        base_name = os.path.splitext(img_name)[0]

        for i in range(split_factor):
            for j in range(split_factor):
                left = j * step_w
                upper = i * step_h
                right = left + step_w
                lower = upper + step_h

                crop = img.crop((left, upper, right, lower))

                crop_name = f"{base_name}_crop{crop_idx}.png"
                crop_path = os.path.join(output_folder, crop_name)
                crop.save(crop_path)

                crop_idx += 1

        print(f"{img_name} -> {crop_idx - 1} crop salvati")

    print(f"\n✅ Crop completato! Immagini salvate in: {output_folder}")


def main(args):
    
    crop_images(args.input_folder, args.num_crops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cropping images in a folder into 4 or 16 subimages")
    parser.add_argument("--input_folder",  type=str, required=True, help="images folder path")
    parser.add_argument("--num_crops", type=int, required=True, choices=[4, 16], help= "number of crop, 4 or 16")

    args = parser.parse_args()
    main(args)