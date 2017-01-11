# A. Lons
# Jan. 2016
#
# DESCRIPTION
# This is the actual file I use to train a CNN for the CIFAR-10 data. To work with a training-opeartion for debugging
# and working through parameter selection, see train_model_testbest.py.

import tensorflow as tf
from tensorflow.python.client import timeline
import time
import read_data as rd
import network_model_0 as nm0

import os

# Make sure to run on second CUDA GPU, which will be number "1", this should force this to occur
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# There is a set number of examples in the CIFAR-10!
NUM_OF_TRAINING_EXAMPLES = 50000

# Files where the training data is stored
FILE_NAMES = ['cifar-10-batches-bin/data_batch_1.bin',
              'cifar-10-batches-bin/data_batch_2.bin',
              'cifar-10-batches-bin/data_batch_3.bin',
              'cifar-10-batches-bin/data_batch_4.bin',
              'cifar-10-batches-bin/data_batch_5.bin']


def run_training(train_filenames, batch_size, n_classes, n_epochs=1):
    """
    DESCRIPTION
    This takes in file-names, batch-sizes, n_classes (10 for CIFAR), and n_epochs and runs. You may have to look at some
    of the hard-coded values internally to adjust the number of preprocesing threads or learning rates.
    :param train_filenames:
    :param batch_size:
    :param n_classes:
    :param n_epochs:
    :return:
    """

    beginning_time_of_run = time.time()

    with tf.Graph().as_default():

        # Get images and labels,
        # Get file names by setting up my readers and queues and pin them to the CPU
        #   see, (https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py)
        with tf.device('/cpu:0'):
            # Note I set the threads here, and the number of epochs, which I add 1 to make sure I do not run out
            # of data.
            images, labels, _ = rd.input_pipline(train_filenames, batch_size=batch_size, numb_pre_threads=6, num_epochs=n_epochs+1, output_type='train')

        # Pin as much to the GPU as possible here: create graph, loss, and optimizer
        with tf.device('/gpu:0'):
            prediction = nm0.generate_Conv_Network(images, batch_size, n_classes, batch_norm=True, is_training=True)
            with tf.name_scope('calc_loss'):
                losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, labels, name='loss_calc_softmax'), name='mean_loss')
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(losses, name='adam_optim_min')

        # Find accuracy
        with tf.device('/gpu:0'):
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    pred_arg_max = tf.argmax(prediction, 1)
                    labl_arg_max = tf.argmax(labels, 1)
                    correct_prediction = tf.equal(pred_arg_max, labl_arg_max)
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Main parts of the operation go here.
        with tf.device('/cpu:0'):

            # Create a session, here I do not want to output anything much to the window so I set "log_de..." to False
            # and I want to myself set GPU vs CPU allocation so I set "allow_sof..." to False.
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

            # Summaries? No, save that for the evaluation or train_model_testbed.py

            # Meta-data or Time-lines? No, save that for the evaluation or train_model_testbed.py

            # Create saver for writing training checkpoints (graph with trainable variables) so I can restore the
            # network some other time.
            saver = tf.train.Saver()

            # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the
            # tf.local_variables_init() because I set the num_epoch in the string producer in the other python file.
            # Tensor Flow r0.12
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='initialize_op')
            # Tensor Flow 0.11
            # init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables(), name='initialize_ops')

            # Run the init operation
            sess.run(init_op)

            # Make a coordinator, to handle all of the threads (mostly the input-pipeline I think)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(n_epochs):

                # Train over all the examples
                num_training_batches = int(NUM_OF_TRAINING_EXAMPLES/batch_size)

                # Track time for batches (this is just to get an idea of the total time I will need to run this
                # method or the evaluation method
                run_batch_times = []

                for j in range(num_training_batches): # The minus one is because of rounding problems I had using int()

                    start_time = time.time()

                    _ = sess.run([optimizer])

                    run_batch_times.append(time.time()-start_time)


                # This is here to help gauge how long everything will take
                print('Epoch ', i)
                avg_run_time = sum(run_batch_times) / float(len(run_batch_times))
                print('  Number of batch runs: ', len(run_batch_times))
                print('  Avg batch run time :', avg_run_time)
                print('  Total epoch run-time:', sum(run_batch_times))

                # Now save the graph!
                path_to_checkpoint = saver.save(sess, 'summaries/chk_pt/model.ckpt', global_step=i)
                print('  Path to check pont: ', path_to_checkpoint)
                print('  Total run time cumm: ', time.time()-beginning_time_of_run, ' (secs)')
                print('  Total run time cumm: ', (time.time()-beginning_time_of_run)/float(60), ' (mins)')

            # Now I have to clean up my threads
            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    batch_size = 500
    n_classes = 10
    n_epochs = 1000
    run_training(FILE_NAMES, batch_size, n_classes, n_epochs)





