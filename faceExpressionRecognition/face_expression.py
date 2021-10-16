from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialExpressionModel(object):
    name_classes = {'1': 'angry',
                    '2': 'disgust',
                    '3': 'fear',
                    '4': "happy",
                    '5': "neutral",
                    "6": 'sad',
                    '7': 'surprise'}

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_weights_file)
        self.preds = []

    def predict_emotion(self, img):
        global session
        set_session(session)
        img = img[np.newaxis, :, :, np.newaxis]
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.name_classes[str(np.argmax(self.preds) + 1)]
