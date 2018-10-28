import os
import tempfile
import random
import unittest
import keras
import numpy as np
from keras_transformer import get_custom_objects, get_model


class TestGetModel(unittest.TestCase):

    def test_save_load(self):
        model = get_model(
            token_num=13,
            embed_dim=30,
            encoder_num=3,
            decoder_num=2,
            head_num=3,
            hidden_dim=120,
            activation='relu',
            dropout_rate=0.05,
            embed_weights=np.random.random((13, 30)),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.categorical_crossentropy,
            metrics={},
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_transformer_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
        try:
            keras.utils.plot_model(model, 'transformer_2_3.png')
        except Exception as e:
            print(e)
        self.assertIsNotNone(model)
