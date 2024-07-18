import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

root = 'data/'

hairline = root + 'Hairline Fracture'
spiral = root + 'Spiral Fracture'
greenstick = root + 'Greenstick fracture'
comminuted = root + 'Comminuted fracture'
dislocation = root + 'Fracture Dislocation'
pathological = root + 'Pathological fracture'
longitudinal = root + 'Longitudinal fracture'
oblique = root + 'Oblique fracture'
impacted = root + 'Impacted fracture'
avulsion = root + 'Avulsion fracture'

folders = np.array(
    [
        hairline,
        spiral,
        greenstick,
        comminuted,
        dislocation,
        pathological,
        longitudinal,
        oblique,
        impacted,
        avulsion
    ]
)

labels = np.array(
    [
        'hairline',
        'spiral',
        'greenstick',
        'comminuted',
        'dislocation',
        'pathological',
        'longitudinal',
        'oblique',
        'impacted',
        'avulsion'
    ]
)

label_ids = np.arange(len(labels))

images_train = []
images_test = []
labels_train = []
labels_test = []
label_ids_train = []
label_ids_test = []

for folder, label, label_id in zip(folders, labels, label_ids):
    for dataset in ("Train", "Test"):
        filepath = os.path.join(folder, dataset)
        for entry in os.listdir(filepath):
            image = cv2.imread(filepath)
            if dataset == "Train":
                images_train.append(image)
                labels_train.append(label)
                label_ids_train.append(label_id)
            else:
                images_test.append(image)
                labels_test.append(label)
                label_ids_test.append(label_id)

images_train, images_val, labels_train, labels_val, label_ids_train, label_ids_val = train_test_split(
    images_train,
    labels_train,
    label_ids_train,
    test_size=0.2,
    random_state=42
)

np.save(root + "images_train.npy", np.array(images_train, dtype=np.float32))
np.save(root + "images_val.npy", np.array(images_val, dtype=np.float32))
np.save(root + "images_test.npy", np.array(images_test, dtype=np.float32))
np.save(root + "labels_train.npy", np.array(labels_train, dtype=np.str_))
np.save(root + "labels_val.npy", np.array(labels_val, dtype=np.str_))
np.save(root + "labels_test.npy", np.array(labels_test, dtype=np.str_))
np.save(root + "label_ids_train.npy", np.array(label_ids_train, dtype=np.int32))
np.save(root + "label_ids_val.npy", np.array(label_ids_val, dtype=np.int32))
np.save(root + "label_ids_test.npy", np.array(label_ids_test, dtype=np.int32))
