# from rembg import remove
# from PIL import Image
# import numpy as np
# import cv2
# import os
# from keras.preprocessing.image import ImageDataGenerator

# # Define input and output paths
# input_path = 'C:/Users/venus/OneDrive/Desktop/deep learning/dataset/Alstonia Scholaris (P2)/0003_0001.JPG'
# output_path = 'output.png'

# # Remove background
# input_image = Image.open(input_path)
# output_image = remove(input_image)
# output_image.save(output_path)

# # Resize the image into 500x500
# output_image = output_image.resize((500, 500))

# # Data augmentation for single leaf color image
# def data_augmentation(image):
#     # Convert PIL image to numpy array
#     img_array = np.array(image)

#     # Expand dimensions to match the shape required by ImageDataGenerator
#     img_array = np.expand_dims(img_array, axis=0)

#     # Create an ImageDataGenerator instance for augmentation
#     datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

#     # Generate augmented images
#     augmented_images = []
#     for batch in datagen.flow(img_array, batch_size=1):
#         augmented_images.append(batch[0].astype('uint8'))
#         if len(augmented_images) >= 5:  # Generate 5 augmented images
#             break

#     # Convert augmented images back to PIL format
#     augmented_images = [Image.fromarray(image) for image in augmented_images]

#     return augmented_images

# # Perform data augmentation on the resized image
# augmented_images = data_augmentation(output_image)

# # Save augmented images
# output_dir = 'augmented_images'
# os.makedirs(output_dir, exist_ok=True)
# for i, image in enumerate(augmented_images):
#     image.save(os.path.join(output_dir, f'augmented_image_{i}.png'))

from rembg import remove
from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor

# Define input and output paths
input_directory = 'C:/Users/venus/OneDrive/Desktop/deep learning/dataset'
output_directory = 'preprocessed_dataset'
output_size = (500, 500)

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Data augmentation for images
def data_augmentation(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Expand dimensions to match the shape required by ImageDataGenerator
    img_array = np.expand_dims(img_array, axis=0)

    # Create an ImageDataGenerator instance for augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Generate augmented images
    augmented_images = []
    for batch in datagen.flow(img_array, batch_size=1):
        augmented_images.append(batch[0].astype('uint8'))
        if len(augmented_images) >= 5:  # Generate 5 augmented images
            break

    # Convert augmented images back to PIL format
    augmented_images = [Image.fromarray(image) for image in augmented_images]

    return augmented_images

# Function to process a single image
def process_image(image_path, output_path):
    # Remove background
    input_image = Image.open(image_path)
    output_image = remove(input_image)

    # Convert image to RGB mode if it's RGBA
    if output_image.mode == 'RGBA':
        output_image = output_image.convert('RGB')

    # Resize the image
    output_image = output_image.resize(output_size)

    # Perform data augmentation
    augmented_images = data_augmentation(output_image)

    # Save augmented images
    for i, augmented_image in enumerate(augmented_images):
        augmented_image.save(os.path.join(output_path, f'augmented_{i}_{os.path.basename(image_path)}'))

    # Save original image without augmentation
    output_image.save(os.path.join(output_path, os.path.basename(image_path)))

# Iterate over each class directory
class_directories = os.listdir(input_directory)
for class_directory in class_directories:
    class_path = os.path.join(input_directory, class_directory)
    output_class_directory = os.path.join(output_directory, class_directory)
    os.makedirs(output_class_directory, exist_ok=True)

    # Process images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        images = [os.path.join(class_path, image_name) for image_name in os.listdir(class_path)]
        for image_path in images:
            executor.submit(process_image, image_path, output_class_directory)
