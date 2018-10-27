import keras
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_position_wise_feed_forward import FeedForward


def _wrap_layer(name, input_layer, build_func, dropout_rate=0.0):
    """Wrap layers with residual, normalization and dropout.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param dropout_rate: Dropout rate.
    :return: Output layer.
    """
    build_output = build_func(input_layer)
    normal_layer = LayerNormalization(name='%s-Norm' % name)(build_output)
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(normal_layer)
    else:
        dropout_layer = normal_layer
    output_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
    return output_layer


def _self_attention_builder(name, head_num, activation):
    """Get multi-head self-attention builder.

    :param name: Prefix of names for internal layers.
    :param head_num: Number of heads in multi-head self-attention.
    :param activation: Activation for multi-head self-attention.
    :return:
    """
    def __self_attention_builder(x):
        return MultiHeadAttention(
            head_num=head_num,
            activation=activation,
            name=name,
        )(x)
    return __self_attention_builder


def _feed_forward_builder(name, hidden_dim, activation):
    """Get position-wise feed-forward layer builder.

    :param name: Prefix of names for internal layers.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for feed-forward layer.
    :return:
    """
    def __feed_forward_builder(x):
        return FeedForward(
            units=hidden_dim,
            activation=activation,
            name=name,
        )(x)
    return __feed_forward_builder


def _get_encoder_component(name, input_layer, head_num, hidden_dim, activation='relu', dropout_rate=0.0):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for multi-head self-attention and feed-forward layer.
    :param dropout_rate: Dropout rate.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=_self_attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=activation,
        ),
        dropout_rate=dropout_rate,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=_feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=activation,
        ),
        dropout_rate=dropout_rate,
    )
    return feed_forward_layer
