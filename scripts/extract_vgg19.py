import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import numpy as np
from keras.applications import VGG19

print(f"Extracting {config.vgg_name} Features...")

model = VGG19(weights=config.vgg_weights,
              include_top=False, 
              input_shape=(config.image_size[0], config.image_size[1], 3))

for file in os.listdir(config.images_folder):
    if config.images_prefix in file:
        file_path = os.path.join(config.images_folder, file)
        images = np.load(file_path, allow_pickle=True)
        
        extracted_features = []
        for image in images:
            feature_image = model.predict(
                image.reshape((1, config.image_size[0], 
                               config.image_size[1], 3)))
            extracted_features.append(feature_image)

        if extracted_features:
                    extracted_features = np.array(extracted_features, dtype=np.int32)
                    new_file_name = file.replace(config.images_prefix, config.vgg_prefix, 1)
                    new_file_path = os.path.join(config.vgg_folder, new_file_name)
                    np.save(new_file_path, extracted_features)
                    print(f'    Processed {file}')
        else:
            print(f'\nNo valid images to process in {file}')

print(f'\n{config.vgg_name} feature extraction completed.')