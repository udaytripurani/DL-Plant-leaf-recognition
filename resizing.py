import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

def resize_image(args):
    img_path, output_path, new_size = args
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, new_size)
    cv2.imwrite(output_path, resized_img)

def resize_images(dataset_dir, output_dir, new_size):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through the dataset directory
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            output_class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)

            # Prepare arguments for parallel processing
            args_list = []
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                if os.path.isfile(img_path):
                    output_path = os.path.join(output_class_dir, filename)
                    args_list.append((img_path, output_path, new_size))

            # Parallel processing
            with Pool(cpu_count()) as pool:
                pool.map(resize_image, args_list)

def main():
    dataset_dir = "C:/Users/venus/OneDrive/Desktop/deep learning/preprocessed_dataset"
    output_dir = "C:/Users/venus/OneDrive/Desktop/deep learning/preprocessed"
    new_size = (64, 64)  # New size for the images

    resize_images(dataset_dir, output_dir, new_size)

if __name__ == "__main__":
    main()
