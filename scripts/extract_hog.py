import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import numpy as np
from skimage import exposure
from skimage.feature import hog

print(f"Extracting {config.hog_name} Features...")

for file in os.listdir(config.images_folder):
    if config.images_prefix in file:
        file_path = os.path.join(config.images_folder, file)
        images = np.load(file_path, allow_pickle=True)
        
        extracted_features = []
        for image in images:
            _, feature_image = hog(
                   image,
                   orientations=9,
                   pixels_per_cell=config.pixels_per_cell,
                   cells_per_block=config.cells_per_block,
                   visualize=True,
                   channel_axis=None
            )
            feature_image = exposure.rescale_intensity(
                   feature_image,
                   in_range=config.rescale_in_range
            )
            extracted_features.append(feature_image)

        if extracted_features:
                    extracted_features = np.array(extracted_features, dtype=np.int32)
                    new_file_name = file.replace(config.images_prefix, config.hog_prefix, 1)
                    new_file_path = os.path.join(config.hog_folder, new_file_name)
                    np.save(new_file_path, extracted_features)
                    print(f'    Processed {file}')
        else:
            print(f'\nNo valid images to process in {file}')

print(f'\n{config.hog_name} feature extraction completed.')
