import unittest
from keras_transformer.backend import keras
from keras_transformer import get_encoders, get_decoders


class TestGetDecoderComponent(unittest.TestCase):

    def test_sample(self):
        encoder_input_layer = keras.layers.Input(shape=(512, 768), name='Encoder-Input')
        decoder_input_layer = keras.layers.Input(shape=(512, 768), name='Decoder-Input')
        encoded_layer = get_encoders(
            encoder_num=2,
            input_layer=encoder_input_layer,
            head_num=12,
            hidden_dim=3072,
            dropout_rate=0.0,
        )
        output_layer = get_decoders(
            decoder_num=2,
            input_layer=decoder_input_layer,
            encoded_layer=encoded_layer,
            head_num=12,
            hidden_dim=3072,
            dropout_rate=0.0,
        )
        model = keras.models.Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model.summary(line_length=160)

        output_layer = get_decoders(
            decoder_num=2,
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
