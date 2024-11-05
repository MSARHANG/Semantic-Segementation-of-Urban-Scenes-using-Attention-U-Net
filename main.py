import os
import data_loader
import model
import train
import utils
import configs


image_list, mask_list = data_loader.load_image_paths(configs.IMAGE_DIR, configs.MASK_DIR)
dataset = data_loader.prepare_dataset(image_list, mask_list, configs.BATCH_SIZE, configs.BUFFER_SIZE)
model = model.attention_unet(input_shape=(128, 128, 3), n_classes=configs.NUM_CLASSES)

history = train.compile_and_train(model, dataset, configs.EPOCHS, configs.BUFFER_SIZE, configs.BATCH_SIZE)

train.show_predictions(model, dataset, num=5)