import zipfile
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np

import shutil
import os

import traceback
import logging

def load_model(path):
  try:
    reconstructed_model = tf.keras.models.load_model(path)
    return reconstructed_model
  except Exception as e:
    logging.error(traceback.format_exc())
    return None

# Loading model
path_2_model = 'YOLOv3-custom-training/model_data/yolo_weights.h5'
reconstructed_model =  load_model(path_2_model)

save_path = 'saved_custom_model'

#reconstructed_model.save(os.path.join(save_path,'custom_model'))


SAVED_MODEL_PATH = 'saved_custom_model/custom_model'
TFLITE_FILE_PATH = 'saved_custom_model/model_lite.tflite'
# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)
