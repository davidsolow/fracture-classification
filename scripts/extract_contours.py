import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import cv2
import numpy as np


print(f"Extracting {config.contours_name} features...")

for file in os.listdir(config.images_folder):
    if config.images_prefix in file:
        file_path = os.path.join(config.images_folder, file)
        images = np.load(file_path, allow_pickle=True)
        
        extracted_features = []
        for image in images:
            _, thresh = cv2.threshold(
                  image, 
                  config.contours_threshold1,
                  config.contours_threshold2,
                  cv2.THRESH_BINARY
                  )
            contours, _ = cv2.findContours(
                  thresh,
                  cv2.RETR_TREE,
                  cv2.CHAIN_APPROX_SIMPLE
                  )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            feature_image = np.zeros_like(image)
            cv2.drawContours(
                  feature_image,
                  contours,
                  -1,
                  (255, 255, 255)
                  )
            extracted_features.append(feature_image)

        if extracted_features:
                    extracted_features = np.array(extracted_features, dtype=np.int32)
                    new_file_name = file.replace(config.images_prefix, config.contours_prefix, 1)
                    new_file_path = os.path.join(config.contours_folder, new_file_name)
                    np.save(new_file_path, extracted_features)
                    print(f'    Processed {file}')
        else:
            print(f'\nNo valid images to process in {file}')

print(f'\n{config.contours_name} extraction completed.')
