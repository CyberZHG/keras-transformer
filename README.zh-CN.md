# Keras Transformer

[![Travis](https://travis-ci.org/CyberZHG/keras-transformer.svg)](https://travis-ci.org/CyberZHG/keras-transformer)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-transformer/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-transformer)
[![Version](https://img.shields.io/pypi/v/keras-transformer.svg)](https://pypi.org/project/keras-transformer/)
![Downloads](https://img.shields.io/pypi/dm/keras-transformer.svg)
![License](https://img.shields.io/pypi/l/keras-transformer.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-theano-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0_beta-blue.svg)

 \[[中文](https://github.com/CyberZHG/keras-transformer/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-transformer/blob/master/README.md)\]

[Transformer](https://arxiv.org/pdf/1706.03762.pdf)的实现。

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

### 翻译

```python
import numpy as np
from keras_transformer import get_model, decode

source_tokens = [
    'i need more power'.split(' '),
    'eat jujube and pill'.split(' '),
]
target_tokens = [
    list('我要更多的抛瓦'),
    list('吃枣💊'),
]

# 生成不同语言的词典
def build_token_dict(token_list):
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

source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

# 添加特殊符号
encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

# 补齐长度
source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))

encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

# 构建和训练模型
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=32,
    encoder_num=2,
    decoder_num=2,
    head_num=4,
    hidden_dim=128,
    dropout_rate=0.05,
    use_same_embed=False,  # 不同语言需要使用不同的词嵌入
)
model.compile('adam', 'sparse_categorical_crossentropy')
model.summary()

model.fit(
    x=[np.array(encode_input * 1024), np.array(decode_input * 1024)],
    y=np.array(decode_output * 1024),
    epochs=10,
    batch_size=32,
)

# 预测过程
decoded = decode(
    model,
    encode_input,
    start_token=target_token_dict['<START>'],
    end_token=target_token_dict['<END>'],
    pad_token=target_token_dict['<PAD>'],
)
print(''.join(map(lambda x: target_token_dict_inv[x], decoded[0][1:-1])))
print(''.join(map(lambda x: target_token_dict_inv[x], decoded[1][1:-1])))
```

### 柱搜索

默认参数下，`decode`只使用概率最高的词作为结果。通过调整`top_k`和`temperature`可以启用柱搜索，较高的温度会使每个词被选中的概率更为平均，而极为接近零的温度相当于`top_k`为1的结果：

```python
decoded = decode(
    model,
    encode_input,
    start_token=target_token_dict['<START>'],
    end_token=target_token_dict['<END>'],
    pad_token=target_token_dict['<PAD>'],
    top_k=10,
    temperature=1.0,
)
print(''.join(map(lambda x: target_token_dict_inv[x], decoded[0][1:-1])))
print(''.join(map(lambda x: target_token_dict_inv[x], decoded[1][1:-1])))
```
