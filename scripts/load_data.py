import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def resize_image(image, target_size):
    """Resize an image while maintaining its aspect ratio."""
    (h, w) = image.shape[:2]
    (target_w, target_h) = target_size
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h
    if aspect_ratio > target_aspect_ratio:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    blank_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    blank_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    return blank_image

print("Loading images...")

images_train = []
images_test = []
labels_train = []
labels_test = []
label_ids_train = []
label_ids_test = []

for folder, label, label_id in zip(config.raw_data_folders, config.labels, config.label_ids):
    for dataset in config.raw_data_subfolders.keys():
        filepath = os.path.join(folder, dataset)
        for entry in os.listdir(filepath):
            full_path = os.path.join(filepath, entry)
            image = cv2.imread(full_path)
            image = resize_image(image, config.image_size)
            if config.raw_data_subfolders[dataset] == "train":
                images_train.append(image)
                labels_train.append(label)
                label_ids_train.append(label_id)
            else:
                images_test.append(image)
                labels_test.append(label)
                label_ids_test.append(label_id)

images_train = np.array(images_train, dtype=np.uint8)
images_test = np.array(images_test, dtype=np.uint8)
labels_train = np.array(labels_train, dtype=np.str_)
labels_test = np.array(labels_test, dtype=np.str_)
label_ids_train = np.array(label_ids_train, dtype=np.int16)
label_ids_test = np.array(label_ids_test, dtype=np.int16)

np.save(config.data_root + "images/images_train.npy", images_train)
np.save(config.data_root + "images/images_test.npy", images_test)
np.save(config.data_root + "images/labels_train.npy", labels_train)
np.save(config.data_root + "images/labels_test.npy", labels_test)
np.save(config.data_root + "images/label_ids_train.npy", label_ids_train)
np.save(config.data_root + "images/label_ids_test.npy", label_ids_test)

print("Images Loaded:")
print(f"    {images_train.shape[0]} train images")
print(f"    {images_test.shape[0]} test images")

print("\nData loading completed.")
