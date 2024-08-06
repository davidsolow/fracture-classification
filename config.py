import numpy as np

# Root data and images folder
data_root = 'data/'
images_folder = data_root + 'images/'

## Load images
# Raw data folders
hairline = data_root + 'raw/Hairline Fracture'
spiral = data_root + 'raw/Spiral Fracture'
greenstick = data_root + 'raw/Greenstick fracture'
comminuted = data_root + 'raw/Comminuted fracture'
dislocation = data_root + 'raw/Fracture Dislocation'
pathological = data_root + 'raw/Pathological fracture'
longitudinal = data_root + 'raw/Longitudinal fracture'
oblique = data_root + 'raw/Oblique fracture'
impacted = data_root + 'raw/Impacted fracture'
avulsion = data_root + 'raw/Avulsion fracture'

raw_data_folders = [
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

# Train/test subfolders
raw_data_subfolders = {
    "Train": "train",
    "Test": "test"
}

# Labels
labels = [
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

label_ids = np.arange(len(labels))

images_prefix = 'images_'

image_size = (512, 512)

val_split = 0.1

# Preprocessing
grayscale_flg = 1
histogram_eq_flg = 1
scaling_flg = 0
resize_shape = (512, 512)
unsharp_mask_flg = 0
unsharp_radius = 2
unsharp_amount = 1

# HOG
hog_name = "HOG"
hog_folder = "data/hog/"
hog_prefix = "hog_"
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
rescale_in_range = (0, 10)

# Canny Edges
canny_edges_name = "Canny Edges"
canny_edges_folder = "data/canny_edges/"
canny_edges_prefix = "canny_edges_"
canny_threshold1 = 100
canny_treshold2 = 200

# Contours
contours_name = "Contours"
contours_folder = "data/contours/"
contours_prefix = "contours_"
contours_threshold1 = 127
contours_threshold2 = 255

# vgg19
vgg19_name = "VGG19"
vgg19_folder = "data/vgg19/"
vgg19_prefix = "vgg19_"
vgg19_weights = 'imagenet'
