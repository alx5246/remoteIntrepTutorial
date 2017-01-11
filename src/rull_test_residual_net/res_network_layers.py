# A. lons
# Decemebr 2016
#
# Here are all the layers that can be put together to form a Res-Net

import tensorflow as tf

def variable_summaries(var, name):
    """
    DESCRIPTION
    Adds a summary to for tensor board visualizations. There is no output, adds to tf.summary under the hood.
    :param var: the variable that I want to analyze, for example, the weights of a convolution layer.
    :return:
    """
    # Remember name_scopes inheret
    with tf.device('/cpu:0'):
        with tf.name_scope('sumry'):
            with tf.name_scope(name):
                with tf.device('/cpu:0'):
                    mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    with tf.device('/cpu:0'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.histogram('histogram', var)


def _variable_on_cpu(name, shape, initializer, trainable=True):
    """
    DESCRIPTION
    Taken from "../cifar10.py", where even though the ops may run on the gpu, it seems they put the variables on the
    CPU.
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for the tf.Variable
    :param trainable: boolean, True means that this variable can be trained or altered by TF's optimizers
    :return: Variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_on_gpu(name, shape, initializer, trainable=True):
    """
    DESCRIPTION
    I have found that striktly the variables onto the gpu is a bit faster, but it might not be in all circumstances
    especially if I am dividing training across GPUs, though
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for the tf.Variable
    :param trainable: boolean, True means that this variable can be trained or altered by TF's optimizers
    :return: Variable object (ie, biases)
    """
    with tf.device('/gpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def batch_normalization_wrapper(inputs, ten_shape, is_training, decay=.999, epsilon=.0000000001):
    """
    DESCRIPTION
    This function is meant to take the batch norm, and switch how this is used from training to evaluation. During
    evaluation we need to use the average values found over training. We assume we are normalizing over the first
    dimension of the inputs!!!!!!
    This is my version of that in http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    Also see https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412 to see how to handle convolution types
    :param inputs: Tensors before going through activation, so simply weights*prior-outputs
    :param ten_shape: integer, depth of input map
    :param is_training: boolean, very important, must be set to False if doing an evaluation
    :param decay: The is for exponential moving average which we use to keep track of the gloabl values of the mean
           and variance
    :return: batch-normed tensor, should be same size as input tensor
    """
    # The variables that will be used during during training to hold mean and var or a particular input batch. These
    # are used only during training epochs.
    bn_scale = _variable_on_gpu('bn_scaling', ten_shape, initializer=tf.constant_initializer(value=1.0, dtype=tf.float32))
    bn_beta = _variable_on_gpu('bn_beta', ten_shape, initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

    # The variables that get updated during learning, and are actually used however in testing. So these are used and
    # updated differently depending on if used suring evaluation or training.
    pop_bn_mean = _variable_on_gpu('pop_bn_mean', ten_shape,
                                   initializer=tf.constant_initializer(value=0.0, dtype=tf.float32),
                                   trainable=False)
    variable_summaries(pop_bn_mean, 'pop_bn_mean')
    pop_bn_var = _variable_on_gpu('pop_bn_var', ten_shape,
                                  initializer=tf.constant_initializer(value=1.0, dtype=tf.float32),
                                  trainable=False)
    variable_summaries(pop_bn_var, 'pop_bn_var')


    if is_training:
        with tf.name_scope('batch_norm_training'):

            # We normalize over depth. Thus if we have a output tensor like [100, 32, 32, 64], we want to normalize over
            # the depth of 64, so we hand the tf.nn.moments a list like [0, 1, 2] to say which dimensions we want to
            # operate over. To calculate this list, we have this simple loop.
            calc_moments_over_which_dimensions = []
            count = 0
            for i in range(len(inputs.get_shape())-1):
                calc_moments_over_which_dimensions.append(count)
                count += 1
            b_mean, b_var = tf.nn.moments(inputs, calc_moments_over_which_dimensions)

            # Track with an exponential moving average. Because I am naming this "train_pop_bn_mean" I have a seperate
            # op that can be run in training, and thus I can still use the variable "pop_bn_mean" later in evaluation.
            train_pop_bn_mean = tf.assign(pop_bn_mean, pop_bn_mean * decay + b_mean * (1 - decay),
                                          name='poulation_mean_calc')
            train_pop_bn_var = tf.assign(pop_bn_var, pop_bn_var * decay + b_var * (1 - decay),
                                         name='poulation_var_calc')

            # Run batch norm (the built in version)
            with tf.control_dependencies([train_pop_bn_mean, train_pop_bn_var]):
                return tf.nn.batch_normalization(inputs, b_mean, b_var, bn_beta, bn_scale, epsilon,
                                                 name='batch_normalization_training')
    else:
        with tf.name_scope('batch_norm_evaluation'):
            return tf.nn.batch_normalization(inputs, pop_bn_mean, pop_bn_var, bn_beta, bn_scale, epsilon,
                                             name='batch_normalization_testing')


def gen_2dconv(input, conv_shape, strides, bias_shape, batch_norm=True, is_training=True):
    """
    DESCRIPTION
    Creates a 2D-convolution operation, where within we will be creating weights, biases, etc. Note we can select this
    to have a batch-norm here, the activation function is not here but rather is outside to make creating Res-Nets
    more like the original paper
    :param input: input tensor
    :param conv_shape: list, [n, m, x, y] where n and m are the kernal sizes, and x is the number of inputs, and y is
           the number of outputs
    :param strides: list [n, m, x, y] see tf.nn.conv2d to see what strides does.
    :param bias_shape: 1D list, should be equal to the number of outputs
    :param batch_norm: boolean, True means we apply a batch_norm before every ReLU
    :param is_training: boolean, needs to be set to true if training, and false is evaluating, this is for setting up
           the batch_norm correctly.
    :return: tensor output of the convolution layer (ReLU(BN(W*A + Bias))) where BN is batch-norm if called for.
    """
    kernel = _variable_on_cpu("weights", conv_shape, initializer=tf.random_normal_initializer())
    variable_summaries(kernel, 'weights')

    biases = _variable_on_cpu("biases", bias_shape, initializer=tf.random_normal_initializer())
    variable_summaries(biases, 'biases')

    conv_op = tf.nn.conv2d(input, kernel, strides=strides, padding='SAME', name='conv2d_op')

    # Apply (or not apply) batch normalization with trainable parameters
    if batch_norm:
        with tf.variable_scope('batch_norm'):
            pre_activ = batch_normalization_wrapper(tf.nn.bias_add(conv_op, biases, name='add_biases_op'), bias_shape, is_training, decay=.999, epsilon=.0000000001)
    else:
        pre_activ = tf.nn.bias_add(conv_op, biases, name='add_biases_op')

    return pre_activ


def res_block(input_tensor, output_depth, down_sample=True, batch_norm=True, is_training=True):
    """

    :param input:
    :param conv_shape:
    :param strides:
    :param bias_shape:
    :param down_sample:
    :param is_training:
    :return:
    """

    # If there is a down-sampling, meaning a max-pooling step, we do this first! The max pooling here is teh standard
    # operation that cuts down the image in half.
    if down_sample:
        with tf.name_scope("max_pooling"):
            input_tensor = tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_op')

    # Now find the input depth, and the output depths, this will be helpful and
    input_depth = input_tensor.get_shape().as_list()[3]

    # Apply the convolution - Relu - Convolutions

    # Apply the first convoltuion
    with tf.name_scope("conv_1"):
        conv_1 = gen_2dconv(input_tensor, [3, 3, input_depth, output_depth], [1, 1, 1, 1], [output_depth],
                            batch_norm=batch_norm, is_training=is_training)

    # Apply the ReLU
    with tf.name_scope("relu_1"):
        relu_1 = tf.nn.relu(conv_1, name='relu_1')

    # Apply the second convolution
    with tf.name_scope("conv_2"):
        conv_2 = gen_2dconv(relu_1, [3, 3, output_depth, output_depth], [1, 1, 1, 1], [output_depth],
                            batch_norm=batch_norm, is_training=is_training)

    # No apply the addtion operator to combine input and output!

    # First we have to pad the if necessary
    if input_depth!=output_depth:
        input_tensor = tf.pad(input_tensor, [[0, 0], [0, 0], [0, 0], [0, output_depth-input_depth]], name='pad_input')

    # Add together
    res_pre_activ= tf.add(conv_2, input_tensor, name='add_block_to_skip')

    # Apply Relu output
    res_output = tf.nn.relu(res_pre_activ, name='relu_output')

    return res_output



