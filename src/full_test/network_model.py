# A. lons
# Decemebr 2016
#
# This is where I make my network models

import tensorflow as tf

def variable_summaries(var):
    """
    DESCRIPTION
    Adds a summary to for tensor board visualizations. There is no output, adds to tf.summary under the hood.
    :param var: the variable that I want to analyze, for example, the weights of a convolutionlayer.
    :return:
    """
    # Remember name_scopes inheret
    with tf.device('/cpu:0'):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('max', tf.reduce_max(var))
            #tf.summary.histogram('histogram', var)


def _variable_on_cpu(name, shape, initializer):
    """
    DESCRIPTION
    Taken from "../cifar10.py", where even though the ops may run on the gpu, it seems they put the variables on the
    CPU.
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for the tf.Variable
    :return: Variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def gen_2dconv(input, conv_shape, strides, bias_shape):
    """
    DESCRIPTION
    Creates a 2D-convolution operation, where within we will be creating both the weights and variables!
    :param input: input tensor
    :param conv_shape: list, [n, m, x, y] where n and m are the kernal sizes, and x is the number of inputs, and y is the number of outputs
    :param strides: list [n, m, x, y] see tf.nn.conv2d to see what strides does.
    :param bias_shape: 1D list, should be equal to the number of outputs
    :param layer_name: string, the name of this layer.
    :return:
    """
    kernel = _variable_on_cpu("weights", conv_shape, initializer=tf.random_normal_initializer())
    variable_summaries(kernel)

    biases = _variable_on_cpu("biases", bias_shape, initializer=tf.random_normal_initializer())
    variable_summaries(biases)

    conv_op = tf.nn.conv2d(input, kernel, strides=strides, padding='SAME', name='conv2d_op')

    add_bias_op = tf.nn.bias_add(conv_op, biases, name='add_biases_op')

    activ = tf.nn.relu(add_bias_op, name='relu_op')

    return activ


def gen_max_pooling(input, kernel_shape, strides):
    """

    :param input:
    :param kernel_shape:
    :param strides:
    :return:
    """
    pool = tf.nn.max_pool(input, ksize=kernel_shape, strides=strides, padding='SAME', name='max_pool_op')
    return pool


def gen_hidden_layer(input, weight_shape, bias_shape):
    """
    DESCRIPTION
    We are assuming all the reshaping has been done outside already!
    :param input: input tensor, ie [batch_size, length of flattened input] <- if the former layer was a conv layer flattened!
    :param kernel_shape: the shape of the weights, ie, [length of flattened input, number of hidden neurons]
    :param bias_shape:
    :return:
    """
    weights = _variable_on_cpu("weights", weight_shape, initializer=tf.random_normal_initializer())
    variable_summaries(weights)

    biases = _variable_on_cpu("biases", bias_shape, initializer=tf.random_normal_initializer())
    variable_summaries(biases)

    output = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, weights, name='mat_mult'), biases, name='add_biases_op'), name='relu_op')

    return output


def gen_output_layer(input, weight_shape, bias_shape):
    """
    DESCRIPTION
    We are assuming all the reshaping has been done outside already!
    :param input: input tensor, ie [batch_size, length of flattened input] <- if the former layer was a conv layer flattened!
    :param kernel_shape: the shape of the weights, ie, [length of flattened input, number of hidden neurons]
    :param bias_shape:
    :return:
    """
    weights = _variable_on_cpu("weights", weight_shape, initializer=tf.random_normal_initializer())
    variable_summaries(weights)

    biases = _variable_on_cpu("biases", bias_shape, initializer=tf.random_normal_initializer())
    variable_summaries(biases)

    output = tf.nn.bias_add(tf.matmul(input, weights, name='mat_mult'), biases, name='add_biases_op')

    return output


def generate_Conv_Network(images, batch_size, n_classes):
    """
    DESCRIPTION
    Building a model of the network
    :param images: the input images
    :param batch_size: the number of images we will have in each batch!
    :param n_classes: the number of output classes we will have.
    :return: the output of the network
    """

    # First convoltuion layer
    with tf.name_scope('layer_1_conv'):
        with tf.variable_scope('layer_1_conv'):
            conv_1 = gen_2dconv(images, [5, 5, 3, 32], [1, 1, 1, 1], [32])

    # First max-pooling operation
    with tf.name_scope('layer_1_pool'):
        pool_1 = gen_max_pooling(conv_1, [1, 2, 2, 1], [1, 2, 2, 1])

    # Second convolution layer
    with tf.name_scope('layer_2_conv'):
        with tf.variable_scope('layer_2_conv'):
            conv_2 = gen_2dconv(pool_1, [5, 5, 32, 64], [1, 1, 1, 1], [64])

    # Second max-pooling layer
    with tf.name_scope('layer_2_pool'):
        pool_2 = gen_max_pooling(conv_2, [1, 2, 2, 1], [1, 2, 2, 1])

    # Now a hidden layer!
    with tf.name_scope('layer_3_hidden'):
        with tf.variable_scope('layer_3_hidden'):
            # We have to reshape here first! We do this by knowning the batch-size, and then using '-1' to automatically
            # select the second dimension. Thus after we reshape we will be left with a tensor of shape [num batches,
            # size of flattened conv output]
            rehaped_conv_output = tf.reshape(pool_2, [batch_size, -1])
            # I am getting the shape of the output, simply following what is done in cifar10.inference(images)
            flattened_dim = rehaped_conv_output.get_shape()[1].value
            hid_3 = gen_hidden_layer(rehaped_conv_output, [flattened_dim, 128], [128])

    # Now add the output layer
    with tf.name_scope('layer_4_output'):
        with tf.variable_scope('layer_4_output'):
            net_output = gen_output_layer(hid_3, [128, n_classes], [n_classes])

    return net_output






