import unittest
import keras
from keras_transformer.transformer import _get_encoder_component, _get_decoder_component


class TestGetDecoderComponent(unittest.TestCase):

    def test_sample(self):
        encoder_input_layer = keras.layers.Input(shape=(512, 768), name='Encoder-Input')
        decoder_input_layer = keras.layers.Input(shape=(512, 768), name='Decoder-Input')
        encoded_layer = _get_encoder_component(
            name='Encoder',
            input_layer=encoder_input_layer,
            head_num=12,
            hidden_dim=3072,
            dropout_rate=0.0,
        )
        output_layer = _get_decoder_component(
            name='Decoder',
            input_layer=decoder_input_layer,
            encoded_layer=encoded_layer,
            head_num=12,
            hidden_dim=3072,
            dropout_rate=0.0,
        )
        model = keras.models.Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model.summary(line_length=160)

        output_layer = _get_decoder_component(
            name='Decoder',
            input_layer=decoder_input_layer,
            encoded_layer=encoded_layer,
            head_num=12,
            hidden_dim=3072,
            dropout_rate=0.1,
        )
        model = keras.models.Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model.summary(line_length=160)
        self.assertIsNotNone(model)
