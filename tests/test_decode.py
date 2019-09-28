import os
import unittest
import numpy as np
from keras_transformer.backend import keras
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
            embed_dim=32,
            encoder_num=3,
            decoder_num=2,
            head_num=4,
            hidden_dim=128,
            dropout_rate=0.05,
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
        )
        model.summary()
        encoder_inputs_no_padding = []
        encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
        for i in range(1, len(tokens)):
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
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path, 'test_transformer.h5')
        if os.path.exists(model_path):
            model.load_weights(model_path, by_name=True)
        else:
            model.fit(
                x=[np.asarray(encoder_inputs * 2048), np.asarray(decoder_inputs * 2048)],
                y=np.asarray(decoder_outputs * 2048),
                epochs=10,
                batch_size=128,
            )
            model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        decoded = decode(
            model,
            encoder_inputs_no_padding * 2,
            start_token=token_dict['<START>'],
            end_token=token_dict['<END>'],
            pad_token=token_dict['<PAD>'],
        )
        token_dict_rev = {v: k for k, v in token_dict.items()}
        for i in range(len(decoded)):
            print(' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1])))
        for i in range(len(decoded)):
            for j in range(len(decoded[i])):
                self.assertEqual(decoder_inputs[i % len(decoder_inputs)][j], decoded[i][j])

        decoded = decode(
            model,
            encoder_inputs_no_padding[2] + [0] * 5,
            start_token=token_dict['<START>'],
            end_token=token_dict['<END>'],
            pad_token=token_dict['<PAD>'],
        )
        for j in range(len(decoded)):
            self.assertEqual(decoder_inputs[2][j], decoded[j], decoded)

        decoded = decode(
            model,
            encoder_inputs_no_padding,
            start_token=token_dict['<START>'],
            end_token=token_dict['<END>'],
            pad_token=token_dict['<PAD>'],
            max_len=4,
        )
        token_dict_rev = {v: k for k, v in token_dict.items()}
        for i in range(len(decoded)):
            print(' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1])))
        for i in range(len(decoded)):
            self.assertTrue(len(decoded[i]) <= 4, decoded[i])
            for j in range(len(decoded[i])):
                self.assertEqual(decoder_inputs[i][j], decoded[i][j], decoded)

        decoded_top_5 = decode(
            model,
            encoder_inputs_no_padding,
            start_token=token_dict['<START>'],
            end_token=token_dict['<END>'],
            pad_token=token_dict['<PAD>'],
            max_len=4,
            top_k=5,
            temperature=1e-10,
        )
        has_diff = False
        for i in range(len(decoded)):
            s1 = ' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1]))
            s5 = ' '.join(map(lambda x: token_dict_rev[x], decoded_top_5[i][1:-1]))
            if s1 != s5:
                has_diff = True
        self.assertFalse(has_diff)

        decoded_top_5 = decode(
            model,
            encoder_inputs_no_padding,
            start_token=token_dict['<START>'],
            end_token=token_dict['<END>'],
            pad_token=token_dict['<PAD>'],
            max_len=4,
            top_k=5,
        )
        has_diff = False
        for i in range(len(decoded)):
            s1 = ' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1]))
            s5 = ' '.join(map(lambda x: token_dict_rev[x], decoded_top_5[i][1:-1]))
            if s1 != s5:
                has_diff = True
        self.assertTrue(has_diff)
