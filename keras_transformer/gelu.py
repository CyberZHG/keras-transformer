import math
from .backend import backend as K


def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + K.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) #save some ops -> math.sqrt(2.0 / math.pi) = 0.7978845608028654
