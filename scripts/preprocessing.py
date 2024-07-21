import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import cv2
import numpy as np
from skimage.filters import unsharp_mask

print('Preprocessing Images...')

def rgb2gray(image):
    if len(image.shape) == 3:
        return np.mean(image, axis=2)
    return image

def scale(image):
    return image * 255

def histEqual(image):
    return cv2.equalizeHist(image)

def unsharp(image, radius, amount):
    return unsharp_mask(image, radius, amount)

image_dir = "data/images/"
for file in os.listdir(image_dir):
    if "images_" in file:
        file_path = os.path.join(image_dir, file)
        images = np.load(file_path, allow_pickle=True)
        
        processed_images = []
        for image in images:
            if config.grayscale_flg:
                image = rgb2gray(image)
            if config.histogram_eq_flg and len(image.shape) == 2:
                image = histEqual(np.array(image, dtype=np.uint8))
            if config.scaling_flg:
                image = scale(image)
            if config.unsharp_mask_flg:
                image = unsharp(image, config.unsharp_radius, config.unsharp_amount)
            processed_images.append(image)

        if processed_images:
            processed_images = np.array(processed_images, dtype=np.uint8)
            np.save(file_path, processed_images)
            print(f'    Processed {file}')
        else:
            print(f'\nNo valid images to process in {file}')

print('\nImage preprocessing completed.')
