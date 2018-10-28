import os
import tempfile
import random
import unittest
import keras
import numpy as np
from keras_transformer import get_custom_objects, get_model, decode


class TestDecode(unittest.TestCase):

    def test_decode(self):
        tokens = 'all work and no play makes jack a dull boy'.split(' ')
        token_dict = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
        }
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
        model = get_model(
            token_num=len(token_dict),
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
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics={},
        )
        model.summary()
        encoder_inputs_no_padding = []
        encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
        for i in range(1, len(tokens) - 1):
            encode_tokens, decode_tokens = tokens[:i], tokens[i:]
            encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(encode_tokens))
            output_tokens = decode_tokens + ['<END>', '<PAD>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
            decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
            encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
            decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
            output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
            encoder_inputs_no_padding.append(encode_tokens[:i + 2])
            encoder_inputs.append(encode_tokens)
            decoder_inputs.append(decode_tokens)
            decoder_outputs.append(output_tokens)
        model.fit(
            x=[np.asarray(encoder_inputs * 1000), np.asarray(decoder_inputs * 1000)],
            y=np.asarray(decoder_outputs * 1000),
            epochs=10,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_transformer_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        decoded = decode(
            model,
            encoder_inputs_no_padding,
            start_token=token_dict['<START>'],
            end_token=token_dict['<END>'],
            pad_token=token_dict['<PAD>'],
        )
        token_dict_rev = {v: k for k, v in token_dict.items()}
        for i in range(len(decoded)):
            print(' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1])))
        for i in range(len(decoded)):
            for j in range(len(decoded[i])):
                self.assertEqual(decoder_inputs[i][j], decoded[i][j], decoded)
        decoded = decode(
            model,
            encoder_inputs_no_padding[2],
            start_token=token_dict['<START>'],
            end_token=token_dict['<END>'],
            pad_token=token_dict['<PAD>'],
        )
        for j in range(len(decoded)):
            self.assertEqual(decoder_inputs[2][j], decoded[j], decoded)
