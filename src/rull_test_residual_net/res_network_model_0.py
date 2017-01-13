# A. Lons
# Jan 2017
#
# DESCRIPTION
# A full resnet model of sorts

import tensorflow as tf
import res_network_layers as rnl

def generate_res_network(images, batch_size, n_classes, batch_norm=True, is_training=True):

    net_layers = []
    net_layers.append(images)

    # No down-sampling in first image please

    layer_1_str = 'first_layer'
    layer_1_depth = 64
    layer_1_num_res = 2

    for i in range(layer_1_num_res):
        var_name = layer_1_str + "_res_%d" % i
        with tf.variable_scope(var_name):
            net_layers.append(rnl.res_block(net_layers[-1], layer_1_depth, down_sample=False, batch_norm=True, is_training=True))


    layer_2_str = 'second_layer'
    layer_2_depth = 128
    layer_2_num_res = 4

    for i in range(layer_2_num_res):
        var_name = layer_2_str + "_res_%d" % i
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    rnl.res_block(net_layers[-1], layer_2_depth, down_sample=True, batch_norm=True, is_training=True))
            else:
                net_layers.append(
                    rnl.res_block(net_layers[-1], layer_2_depth, down_sample=False, batch_norm=True, is_training=True))

    layer_3_str = 'third_layer'
    layer_3_depth = 256
    layer_3_num_res = 6

    for i in range(layer_3_num_res):
        var_name = layer_3_str + "_res_%d" % i
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    rnl.res_block(net_layers[-1], layer_3_depth, down_sample=True, batch_norm=True, is_training=True))
            else:
                net_layers.append(
                    rnl.res_block(net_layers[-1], layer_3_depth, down_sample=False, batch_norm=True, is_training=True))

    layer_4_str = 'fourth_layer'
    layer_4_depth = 512
    layer_4_num_res = 1

    for i in range(layer_4_num_res):
        var_name = layer_4_str + "_res_%d" % i
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    rnl.res_block(net_layers[-1], layer_4_depth, down_sample=True, batch_norm=True, is_training=True))
            else:
                net_layers.append(
                    rnl.res_block(net_layers[-1], layer_4_depth, down_sample=False, batch_norm=True, is_training=True))


    # Now add a fully connected layer please!
    with tf.variable_scope('fully_connected'):
        rehaped_conv_output = tf.reshape(net_layers[-1], [batch_size, -1], name='reshape_conv_output')
        # I am getting the shape of the output, simply following what is done in cifar10.inference(images)
        flattened_dim = rehaped_conv_output.get_shape()[1].value
        net_layers.append(rnl.gen_output_layer(rehaped_conv_output, [flattened_dim, n_classes], [n_classes]))

    return net_layers[-1]


def loss(prediction, labels):

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(prediction, labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='mean_cross_entropy_loss')

    # We do it this way because this is how it was done in multi-gpu CIFAR-10 example
    #tf.add_to_collection("losses", cross_entropy_mean)
    return cross_entropy_mean


