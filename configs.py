import os

PATH = './dataset/'
IMAGE_DIR = os.path.join(PATH, 'CameraRGB/')
MASK_DIR = os.path.join(PATH, 'CameraMask/')
BUFFER_SIZE = 500
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 23
