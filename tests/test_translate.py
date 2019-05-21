# encoding: utf-8
from __future__ import unicode_literals

import unittest
import numpy as np
from keras_transformer import get_model, decode


class TestTranslate(unittest.TestCase):

    @staticmethod
    def _build_token_dict(token_list):
        token_dict = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
        }
        for tokens in token_list:
            for token in tokens:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)
        return token_dict

    def test_translate(self):
        source_tokens = [
            'i need more power'.split(' '),
            'eat jujube and pill'.split(' '),
        ]
        target_tokens = [
            list('我要更多的抛瓦'),
            list('吃枣💊'),
        ]

        # Generate dictionaries
        source_token_dict = self._build_token_dict(source_tokens)
        target_token_dict = self._build_token_dict(target_tokens)
        target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

        # Add special tokens
        encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
        decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
        output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

        # Padding
        source_max_len = max(map(len, encode_tokens))
        target_max_len = max(map(len, decode_tokens))

        encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
        decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
        output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

        encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
        decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
        decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

        # Build & fit model
        model = get_model(
            token_num=max(len(source_token_dict), len(target_token_dict)),
            embed_dim=32,
            encoder_num=2,
            decoder_num=2,
            head_num=4,
            hidden_dim=128,
            dropout_rate=0.05,
            use_same_embed=False,  # Use different embeddings for different languages
        )
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.summary()
        model.fit(
            x=[np.array(encode_input * 1024), np.array(decode_input * 1024)],
            y=np.array(decode_output * 1024),
            epochs=10,
            batch_size=32,
        )

        # Predict
        decoded = decode(
            model,
            encode_input,
            start_token=target_token_dict['<START>'],
            end_token=target_token_dict['<END>'],
            pad_token=target_token_dict['<PAD>'],
        )
        for i in range(len(encode_input)):
            predicted = ''.join(map(lambda x: target_token_dict_inv[x], decoded[i][1:-1]))
            self.assertEqual(''.join(target_tokens[i]), predicted)
