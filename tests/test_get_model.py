import os
import tempfile
import unittest
import numpy as np
from keras_transformer.backend import keras
from keras_transformer import get_custom_objects, get_model


class TestGetModel(unittest.TestCase):

    def test_get_same(self):
        model = get_model(
            token_num=13,
            embed_dim=30,
            encoder_num=3,
            decoder_num=2,
            head_num=3,
            hidden_dim=120,
            attention_activation=None,
            feed_forward_activation='relu',
            dropout_rate=0.05,
            use_same_embed=True,
            embed_weights=np.random.random((13, 30)),
            trainable=False,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.categorical_crossentropy,
            metrics={},
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_transformer_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
        try:
            keras.utils.plot_model(model, 'transformer_same.png')
        except Exception as e:
            print(e)
        self.assertIsNotNone(model)

    def test_get_diff(self):
        model = get_model(
            token_num=[13, 14],
            embed_dim=30,
            encoder_num=3,
            decoder_num=2,
            head_num=3,
            hidden_dim=120,
            attention_activation=None,
            feed_forward_activation='relu',
            dropout_rate=0.05,
            use_same_embed=False,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.categorical_crossentropy,
            metrics={},
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_transformer_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
        try:
            keras.utils.plot_model(model, 'transformer_diff.png')
        except Exception as e:
            print(e)
        self.assertIsNotNone(model)
