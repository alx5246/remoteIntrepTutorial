# A. Lons
# Jan. 2016
#
# DESCRIPTION
# We put the networks in seperate python files, so it is easy to recall different and specific networks we formally
# used in the past.
#
# Network-Model-0 is simply a CNN with batch normalizations.


import tensorflow as tf
import network_layers as nl

def generate_Conv_Network(images, batch_size, n_classes, batch_norm=True, is_training=True, cpu=True, gpu=0):
    """
    DESCRIPTION
    Building a model of the network
    :param images: the input images
    :param batch_size: the number of images we will have in each batch!
    :param n_classes: the number of output classes we will have.
    :return: the output of the network
    """

    # First convoltuion layer
    # I do not need to duplicate scope, tf.variable_scope seesm to act like tf.name_scope when it comes to graph
    #with tf.name_scope('layer_1_conv'):
    with tf.variable_scope('layer_1_conv'):
        conv_1 = nl.gen_2dconv(images, [3, 3, 3, 64], [1, 1, 1, 1], [64], batch_norm=batch_norm, is_training=is_training, cpu=cpu, gpu=gpu)

    with tf.variable_scope('layer_1_1_conv'):
        conv_1_1 = nl.gen_2dconv(conv_1, [3, 3, 64, 64], [1, 1, 1, 1], [64], batch_norm=batch_norm, is_training=is_training, cpu=cpu, gpu=gpu)

    # First max-pooling operation
    with tf.name_scope('layer_1_pool'):
        pool_1 = nl.gen_max_pooling(conv_1_1, [1, 2, 2, 1], [1, 2, 2, 1])

    # Second convolution layer
    #with tf.name_scope('layer_2_conv'):
    with tf.variable_scope('layer_2_conv'):
        conv_2 = nl.gen_2dconv(pool_1, [3, 3, 64, 128], [1, 1, 1, 1], [128], batch_norm=batch_norm, is_training=is_training, cpu=cpu, gpu=gpu)

    with tf.variable_scope('layer_2_1_conv'):
        conv_2_1 = nl.gen_2dconv(conv_2, [3, 3, 128, 128], [1, 1, 1, 1], [128], batch_norm=batch_norm, is_training=is_training, cpu=cpu, gpu=gpu)

    with tf.variable_scope('layer_2_2_conv'):
        conv_2_2 = nl.gen_2dconv(conv_2_1, [3, 3, 128, 128], [1, 1, 1, 1], [128], batch_norm=batch_norm, is_training=is_training, cpu=cpu, gpu=gpu)

    # Second max-pooling layer
    with tf.name_scope('layer_2_pool'):
        pool_2 = nl.gen_max_pooling(conv_2_2, [1, 2, 2, 1], [1, 2, 2, 1])

    # Now a hidden layer!
    #with tf.name_scope('layer_3_hidden'):
    with tf.variable_scope('layer_3_hidden'):
        # We have to reshape here first! We do this by knowning the batch-size, and then using '-1' to automatically
        # select the second dimension. Thus after we reshape we will be left with a tensor of shape [num batches,
        # size of flattened conv output]
        rehaped_conv_output = tf.reshape(pool_2, [batch_size, -1], name='rehape')
        # I am getting the shape of the output, simply following what is done in cifar10.inference(images)
        flattened_dim = rehaped_conv_output.get_shape()[1].value
        hid_3 = nl.gen_hidden_layer(rehaped_conv_output, [flattened_dim, 512], [512], batch_norm=batch_norm, is_training=is_training, cpu=cpu, gpu=gpu)

    with tf.variable_scope('layer_4_hidden'):
        hid_4 = nl.gen_hidden_layer(hid_3, [512, 256], [256], batch_norm=batch_norm, is_training=is_training, cpu=cpu, gpu=gpu)

    # Now add the output layer
    #with tf.name_scope('layer_4_output'):
    with tf.variable_scope('layer_5_output'):
        net_output = nl.gen_output_layer(hid_4, [256, n_classes], [n_classes])

    return net_output


def loss(prediction, labels):

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(prediction, labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='mean_cross_entropy_loss')
    tf.add_to_collection("losses", cross_entropy_mean)
