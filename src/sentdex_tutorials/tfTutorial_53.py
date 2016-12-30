# A. Lons
# December 2016
#
# I am doing the tutorials from sendex on youtube, in particular this is video #53 from sentdex series "Machine Learning
# with Python" and is named "Reccurent Neural Networks (RNN) - Deep Learning with Neural Networks and TensorFlow 10",
# and also a second video #54 from sentdex series named, "RNN Example in Tensorflow - Deep Learning with Neural
# Networks"
#
# Because I am running over remote, I had some trouble with the download of files (mnist), see below.
#
# IN these two tutorial we are tyring to use GPU compute and train an RNN (LSTM)


import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data # This is where we will get our example data.

# (1) Load in the data (somehow using built in tensor flow)
# "one_hot" means 1 is on and the rest are off (in circiuts), we have 10 classes, and calling this "one_hot" is to
# be such that we have 10 outputs (one for each class). We need to make sure we can load all this in. For other
# dataset we need to be smart about how to load in based on size of ram.
# (2) I originally have had some problems. Originally I had code like, .
#   mnist = input_data.read_data_sets("/home/alex/pythonCode/tensorFlowTutorials/src/tutorial_1/", one_hot=True)
# however, that gave me errors. I think it is because the data needs to be uploaded first? So I downloaded the data
# seperately and dumped it into the main folder (t10k-images-idx-ubyte.gz, etc.). Then because I am not sure how
# the paths work given I will be working over remote, I chagnes the path to just "src/" because that exists on
# the remote as well.
mnist = input_data.read_data_sets("src/", one_hot=True)

# Setup epochs
hm_epochs = 10
# Number of classes
n_classes = 10
# Batch size, this is how many inputs we give at once.
batch_size = 128

#Chu
chunk_size = 28
n_chunks = 28
rnn_size = 128

# Define some placeholders, x will be input (flatten). Notes I think the tf.placeholder still constitutes an 'op'
x = tf.placeholder('float', [None, n_chunks, chunk_size]) # Flatten input images.
y = tf.placeholder('float') # This will be the label

# Now define neural network model, this is essentially the computation graph (tha majority of it)
def reccurrent_network_model(x):

    # NOTE, we can call variables from outside scope of the method! This is somthing I do not think of regularily!

    # Single layer of the RNN neurons
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    # We are doing this because we need the data to be in a particualr form for tf's rnn_cell
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output

# Now we have modeled the neural network! This means we are basically done with the computational graph. We now have to
# define some other stuff and what to do with the model.

# A method to train
def train_neural_network(x, y):

    # We are saying now that the prediction is running the former method above, that is we use our former method to
    # define the model
    prediction = reccurrent_network_model(x)

    # Now we generate a cost function (so tf knows what this is)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # No we have make an optimizer (adam default learning_rate = 0.001), NOTE there are other training methods
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Run tf 'session'
    with tf.Session() as sess:

        # We have to initialize the variables (all those that we have formally created, all created are part of the
        # basic graph)
        sess.run(tf.initialize_all_variables())

        # Run over the epochs
        for epoch in range(hm_epochs):

            epoch_loss = 0.

            # We are going to use a specified batch size.
            for _ in range(int(mnist.train.num_examples/batch_size)):

                # This magic command below automaticllay goes through data of specific batch size.
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # We have to resahpe
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))


                # use session run function! to call optimizer
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))

train_neural_network(x, y)