import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import cv2
import numpy as np

print(f"Extracting {config.canny_edges_name} features...")

for file in os.listdir(config.images_folder):
    if config.images_prefix in file:
        file_path = os.path.join(config.images_folder, file)
        images = np.load(file_path, allow_pickle=True)
        
        extracted_features = []
        for image in images:
            feature_image = cv2.Canny(
                  image,
                  threshold1=config.canny_threshold1,
                  threshold2=config.canny_treshold2
                  )
            extracted_features.append(feature_image)

        if extracted_features:
                    extracted_features = np.array(extracted_features, dtype=np.int32)
                    new_file_name = file.replace(config.images_prefix, config.canny_edges_prefix, 1)
                    new_file_path = os.path.join(config.canny_edges_folder, new_file_name)
                    np.save(new_file_path, extracted_features)
                    print(f'    Processed {file}')
        else:
            print(f'\nNo valid images to process in {file}')

print(f'\n{config.canny_edges_name} extraction completed.')
