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
        return activate(tf.matmul(input_, weights) + biases, type_=activation)


def conv2d(input_, filters, kernel_size=(3, 3), strides=(1, 1),
           padding='SAME', activation='linear', name=None):
    input_channel = input_.get_shape().as_list()[-1]
    if name is None:
        name = 'conv2d_{}'.format(filters)
    with tf.variable_scope(name):
        weights = tf.Variable(
            tf.truncated_normal([kernel_size[1], kernel_size[0], input_channel, filters],
                                stddev=0.1),
            name=name+'weights')
        biases = tf.Variable(tf.zeros([filters]), name=name+'biases')
        return activate(tf.nn.conv2d(input_, weights,
                                     strides=(1, strides[1], strides[0], 1), padding=padding) + biases,
                        type_=activation)


def pool2d(input_, kernel_size, mode='max', strides=(2, 2),
           padding='SAME', name=None):
    if mode == 'max' or 'Max' or 'MAX':
        return tf.nn.max_pool(input_,
                              ksize=[1, kernel_size[1], kernel_size[0], 1],
                              strides=[1, strides[1], strides[0], 1],
                              padding=padding,
                              name=name)
    elif mode == 'average' or 'Average' or 'AVERAGE':
        return tf.nn.avg_pool(input_,
                              ksize=[1, kernel_size[1], kernel_size[0], 1],
                              strides=[1, strides[1], strides[0], 1],
                              padding=padding,
                              name=name)
    else:
        print("Pooling mode : {} is not implemented...".format(mode))
        exit()


def activate(input_, type_):
    if type_ == 'relu' or 'ReLU':
        return tf.nn.relu(input_, name='relu')
    elif type_ == 'sigmoid':
        return tf.sigmoid(input_, name='sigmoid')
    elif type_ == 'linear' or 'Linear' or 'LINEAR':
        return input_
    else:
        print("Activation : {} is not implemented...".format(type_))
        exit()
