import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import numpy as np
from sklearn.decomposition import PCA

print(f"Extracting {config.pca_name} Features...")

X_train = np.load('data/images/images_train.npy')
n, h, w = X_train.shape
X_train = X_train.reshape(n, h * w)

n_components = config.n_components
pca = PCA(n_components=n_components)
pca.fit(X_train)

for file in os.listdir(config.images_folder):
    if config.images_prefix in file:
        file_path = os.path.join(config.images_folder, file)
        images = np.load(file_path, allow_pickle=True)
        
        extracted_features = []
        for image in images:
            reshaped = image.reshape(1, -1)
            feature_image = pca.transform(reshaped)
            feature_image = feature_image.flatten()
            extracted_features.append(feature_image)

        if extracted_features:
                    extracted_features = np.array(extracted_features, dtype=np.int32)
                    new_file_name = file.replace(config.images_prefix, config.pca_prefix, 1)
                    new_file_path = os.path.join(config.pca_folder, new_file_name)
                    np.save(new_file_path, extracted_features)
                    print(f'    Processed {file}')
        else:
            print(f'\nNo valid images to process in {file}')

print(f'\n{config.pca_name} feature extraction completed.')
