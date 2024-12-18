# ---------------------------
# Configuration
# ---------------------------

RANDOM_SEED = 42

TRAIN_IMAGES_DIR = 'data/training/images'
TRAIN_GROUNDTRUTH_DIR = 'data/training/groundtruth'
PATCHES_IMG_DIR = 'data/training/patches/images'
PATCHES_GROUNDTRUTH_DIR = 'data/training/patches/groundtruth'

TEST_IMAGES_DIR = 'data/test_set_images'
CUSTOM_TEST_IMAGES_DIR = 'data/custom_test_images'
PREDICTED_GROUNDTRUTH_DIR = 'data/predicted_masks'

MODEL_WEIGHTS_PATH = 'saved_models/unet.keras'

NUM_TRAIN_IMAGES = 100
TRAIN_IMG_HEIGHT = 400
TRAIN_IMG_WIDTH = 400
TRAIN_IMG_CHANNELS = 3

PATCH_SIZE = 256
TEST_IMAGE_SIZE = 608

NUM_RANDOM_ANGLES = 4
SET_RANDOM_ANGLES = False

TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 4
