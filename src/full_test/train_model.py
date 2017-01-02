# A. lons
# Decemebr 2016
#
# This is where we run the whole thing

import tensorflow as tf
from tensorflow.python.client import timeline
import read_data as rd
import network_model as nm

# There is a set number of examples in the CIFAR-10
NUM_OF_TRAINING_EXAMPLES = 60000


def run_training(filenames, batch_size, n_classes, n_epochs):

    # Get file names by setting up my readers and queues and pin them to the CPU
    #   see, (https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py)
    #   in method, inputs(), I think I can "Force all teh input processing onto the CPU" by calling the tf.device here
    #   as long as I have "allow_soft_placement=False" in the session settings.
    with tf.device('/cpu:0'):
        images, labels, _ = rd.input_pipline(filenames, batch_size=batch_size, numb_pre_threads=4, num_epochs=10)

    with tf.device('/gpu:0'):
        # Create the network graph
        prediction = nm.generate_Conv_Network(images, batch_size, n_classes)
        # Now we generate a cost function (so tf knows what this is)
        losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, labels))
        # Now generate optimizer!
        optimizer = tf.train.AdamOptimizer().minimize(losses)

    with tf.device('/cpu:0'):
        # Find accuracy
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.device('/cpu:0'):

        # Create a session, this is done in how-to and cifar-10 example (in the cifar-10 the also have some configs).
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

        # Now prepare all summaries
        # mergedSummaries = tf.summary.merge_all() # <- these work in newer versions of TF
        # trian_writer = tf.summary.FileWriter('trainingsum/train_summary', sess.graph) # # <- these work in newer versions of TF r0.12
        # For the tf 0.11 version of TF, I have to have the following
        merged = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('summaries/summary', sess.graph)

        # I need to run meta-data which will help for 'time-lines' and if I want to output more info
        #   a) To get a time-line to work see running meta-data see http://stackoverflow.com/questions/40190510/tensorflow-
        #   how-to-log-gpu-memory-vram-utilization/40197094
        #   b) To get detailed run information to text file see http://stackoverflow.com/questions/40190510/tensorflow-how-t
        #   o-log-gpu-memory-vram-utilization/40197094
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the tf.local_variables_init()
        # because I set the num_epoch in the string producer in the other python file.
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # <- this works for newer version of TF r0.12.
        # In tf 0.11 I have to use older functions
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables(), name='initialize_ops')

        # Run the init operation
        sess.run(init_op)

        # Make a coordinator,
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run training for a specific number of training examples.
        counter = 0
        for i in range(n_epochs):

            # Train over 90% of examples, save the others for testing
            #num_training_batches = int(NUM_OF_TRAINING_EXAMPLES/batch_size) - 1
            num_training_batches = 10
            for j in range(num_training_batches):

                summary, _ = sess.run([merged, optimizer], options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(summary, counter)
                counter += 1

            # Test over last batch!
            summary, acc = sess.run([merged, accuracy], options=run_options, run_metadata=run_metadata)

            print('Accuracy:', acc)

        # with open("meta_data_run.txt", "w") as out:
        #    out.write(str(run_metadata))
        t1 = timeline.Timeline(run_metadata.step_stats)
        ctf = t1.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

        # Now I have to clean up
        summary_writer.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()


filenames = ['cifar-10-batches-bin/data_batch_1.bin', 'cifar-10-batches-bin/data_batch_2.bin']
batch_size = 128
n_classes = 10
n_epochs = 25

run_training(filenames, batch_size, n_classes, n_epochs)



