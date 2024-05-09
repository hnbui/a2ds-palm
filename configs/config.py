import torch
import os

# base dataset path
DATASET_PATH = 'data/'

# determine paths to image and mask datasets
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, 'Classification_resize/')
MASK_DATASET_PATH = os.path.join(DATASET_PATH, 'Annotation/Disc_Masks_resize/')

# determine training device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# pin memory during data loading to GPU
PIN_MEMORY = True if DEVICE == 'cuda' else False

# define dataset's characteristics for UNet
NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 3

# define training parameters
INIT_LR = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 4

# base output path
OUTPUT = 'results/'

# paths to save model, plot, and test results
MODEL_PATH = os.path.join(OUTPUT, "a2ds_palm_unet.pth")
PLOT_PATH = os.path.sep.join([OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([OUTPUT, "test_paths.txt"])

