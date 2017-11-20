
import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)
