# A. Lons
# December 2016
#
# I am doing the tutorials from sendex on youtube, in particular this is video #55 from sentdex series "Machine Learning
# with Python" and is named "Convolutional Neural Network Basics - Deep Learning with TensorFlow 12" and a second video
# #56 named "Convolutional Neural Networks with TensorFlow - Deep Learning with Neural Networks 13"
#
# Again we take code from his former tutorials and modify it.
#
# We start with a vanilla CNN, and then later we add Drop-out (most popular with convnet)
#
# I have modifications to log things as well, and try to pin different operations to the gpu vs cpu

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # This is where we will get our example data.

# I cannot see to just load in data as before,
#   mnist = input_data.read_data_sets("/home/alex/pythonCode/tensorFlowTutorials/src/tutorial_1/", one_hot=True)
# as I am using a remote comptuer. To counter this I download the files (command line) onto the remote and changed
# the path so that it could refer to the remote's path (or locals as the last folders have the same name)
mnist = input_data.read_data_sets("src/", one_hot=True)

# Number of classes
n_classes = 10
# Batch size, this is how many inputs we give at once.
batch_size = 1024

# Define some placeholders, x will be input (flatten). Notes I think the tf.placeholder still constitutes an 'op'
x = tf.placeholder('float', [None, 784], name='cnn_input') # Flatten input images.
y = tf.placeholder('float', name='cnn_target') # This will be the label

# This is for drop out
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


# As done in TF demo, we are going to make a function to define the way in which the convolutions occur. TF
# has some nice functions built in for this purpose as well.
def conv2d(x, W, layer_name):
    with tf.name_scope(layer_name):
        conv_op = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name="conv2d_op")
    return conv_op


# As done in TF demo, we are going to make as function to define the way in which the pooling occurs!. TF has some
# nice built in functions for this purpose as well which we here use.
def maxpool2d(x, layer_name):
    with tf.name_scope(layer_name):
    # ksize is the size of the pooling window, and the strides move so no overlap
        pool_op = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="max_pool_op")
    return pool_op


def convolutional_neural_network_model(x):


    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),    # 5X5 features, for 1 input, and 32 ouputs
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),   # 5X5 features, for 32 input, and 64 ouputs
               'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),      # Fully connected to 1024 neurons               #
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases  = {'b_conv1': tf.Variable(tf.random_normal([32])),
               'b_conv2': tf.Variable(tf.random_normal([64])),
               'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    # Handle data, take 1D and turn into a an image, (the tf syntax is very similar to numpy)
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Call our convolution and pooling
    conv1 = conv2d(x, weights['W_conv1'], 'conv_layer_1')
    conv1 = maxpool2d(conv1, 'max_pool_layer_1')

    conv2 = conv2d(conv1, weights['W_conv2'], 'conv_layer_2')
    conv2 = maxpool2d(conv2, 'max_pool_layer_2')

    # Now fully connected
    fc = tf.reshape(conv2, [-1, 7*7*64], name='reshape_conv_output')
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'], name='relu_output')

    # We add drop-out here for the fully connected
    fc = tf.nn.dropout(fc, keep_rate)

    # OUtput!
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

# Now we have modeled the neural network! This means we are basically done with the computational graph. We now have to
# define some other stuff and what to do with the model.

# A method to train
def train_neural_network(x, y):

    # We are saying now that the prediction is running the former method above, that is we use our former method to
    # define the model
    prediction = convolutional_neural_network_model(x)

    # Now we generate a cost function (so tf knows what this is)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # No we have make an optimizer (adam default learning_rate = 0.001), NOTE there are other training methods
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Setup epochs
    hm_epochs = 2

    # Run tf 'session', if I set log to true I can see what operations are going where.
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)) as sess:
    #with tf.Session() as sess:

        # (AJL) addition of cpu specific
        with tf.device('/cpu:0'):

            # We have to initialize the variables (all those that we have formally created, all created are part of the
            # basic graph)
            sess.run(tf.initialize_all_variables())

            # Run over the epochs
            for epoch in range(hm_epochs):

                epoch_loss = 0.

                # We are going to use a specified batch size.
                for _ in range(int(mnist.train.num_examples/batch_size)):

                    #(AJL) addition of cpu specific
                    #with tf.device('/cpu:0'):
                    # This magic command below automaticllay goes through data of specific batch size.
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                    with tf.device('gpu:0'):
                        # use session run function! to call optimizer
                        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                # (AJL) is testing size killing my run? Lets see by batching the test will correct for this?
                test_batch_size = 1024
                with tf.device('/cpu:0'):
                    #batch the test
                    for _ in range(int(mnist.test.num_examples/test_batch_size)):
                        test_x, test_y = mnist.test.next_batch(test_batch_size)
                        with tf.device('/gpu:0'):
                            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x, y)