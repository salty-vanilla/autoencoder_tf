import tensorflow as tf
import math


def dense(input_, input_dim, output_dim, activation='linear', name=None):
    if name is None:
        name = 'dense_{}'.format(output_dim)
    with tf.variable_scope(name):
        weights = tf.Variable(
            tf.truncated_normal([input_dim, output_dim],
                                stddev=1.0 / math.sqrt(float(input_dim))),
            name=name+'weights')
        biases = tf.Variable(tf.zeros([output_dim]), name=name+'biases')

        if activation == 'relu' or 'ReLU':
            return tf.nn.relu(tf.matmul(input_, weights) + biases, name='relu')
        elif activation == 'sigmoid':
            return tf.sigmoid(tf.matmul(input_, weights) + biases, name='sigmoid')
        elif activation == 'linear' or 'Linear' or 'LINEAR':
            return tf.matmul(input_, weights) + biases

