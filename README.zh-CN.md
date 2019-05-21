# Keras Transformer

[![Travis](https://travis-ci.org/CyberZHG/keras-transformer.svg)](https://travis-ci.org/CyberZHG/keras-transformer)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-transformer/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-transformer)
[![Version](https://img.shields.io/pypi/v/keras-transformer.svg)](https://pypi.org/project/keras-transformer/)

[Transformer](https://arxiv.org/pdf/1706.03762.pdf)实现。

## 安装

```bash
pip install keras-transformer
```

## 使用

### 训练

```python
import numpy as np
from keras_transformer import get_model

# 构建一个toy词典
tokens = 'all work and no play makes jack a dull boy'.split(' ')
token_dict = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
}
for token in tokens:
    if token not in token_dict:
        token_dict[token] = len(token_dict)

# 生成toy数据
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

# 构建模型
model = get_model(
    token_num=len(token_dict),
    embed_dim=30,
    encoder_num=3,
    decoder_num=2,
    head_num=3,
    hidden_dim=120,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    embed_weights=np.random.random((13, 30)),
)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.summary()

# Train the model
model.fit(
    x=[np.asarray(encoder_inputs * 1000), np.asarray(decoder_inputs * 1000)],
    y=np.asarray(decoder_outputs * 1000),
    epochs=5,
)
```

### 预测

```python
from keras_transformer import decode

decoded = decode(
    model,
    encoder_inputs_no_padding,
    start_token=token_dict['<START>'],
    end_token=token_dict['<END>'],
    pad_token=token_dict['<PAD>'],
    max_len=100,
)
token_dict_rev = {v: k for k, v in token_dict.items()}
for i in range(len(decoded)):
    print(' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1])))
```
