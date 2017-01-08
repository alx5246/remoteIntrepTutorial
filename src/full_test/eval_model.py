# A. lons
# Decemebr 2016
#
# This is where we run the whole thing

import tensorflow as tf
from tensorflow.python.client import timeline
import read_data as rd
import network_model_0 as nm0
import os
import time

# Make sure we use the right device, in this case I want to use the first GPU, GPU 0 as I am training with GPU 1
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# There is a set number of examples in the CIFAR-10
NUM_OF_TRAINING_EXAMPLES = 10000

# The file names of what we want to test
FILE_NAMES = ['cifar-10-batches-bin/test_batch.bin']


def run_training(test_filenames, batch_size, n_classes, delay_time, n_epochs=1):

    with tf.Graph().as_default() as g:

        # Get images and labels,
        # Get file names by setting up my readers and queues and pin them to the CPU. Note I do not many threads here
        # becaus this is just the evaluation graph so its speed is not as important.
        #   see, (https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py)
        with tf.device('/cpu:0'):
            images, labels, key = rd.input_pipline(test_filenames, batch_size=batch_size, numb_pre_threads=2, num_epochs=n_epochs+1, output_type='test')

        with tf.device('/gpu:0'):
            # Create the network graph
            prediction = nm0.generate_Conv_Network(images, batch_size, n_classes, batch_norm=True, is_training=False)

            with tf.name_scope('accuracy'):
                # Find accuracy, I think I can run these on the GPU
                pred_arg_max = tf.argmax(prediction, 1, name='find_which_class_pred')
                labl_arg_max = tf.argmax(labels, 1, name='find_which_class_labl')
                correct_prediction = tf.equal(pred_arg_max, labl_arg_max, name='find_correct_prediction')
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='cast_prediction_errors'), name='find_avg_acc')
                # Add a summary object
            with tf.device('/cpu:0'):
                tf.summary.histogram('predicted_labels', pred_arg_max)
                tf.summary.histogram('actual_labels', labl_arg_max)
                tf.summary.scalar('Accuracy', accuracy)

        with tf.device('/cpu:0'):

            # Create saver for writing training checkpoints, using this "trainable_variables" seems to have helped.
            # Howeer once I stated to include batch-norms I had to get rid of that because some of my variables were
            # not trainable.
            #saver = tf.train.Saver(tf.trainable_variables(), name='saver_loader')
            saver = tf.train.Saver( name='saver_loader')

            # Now prepare all summaries (these following lines will be be based on the tensorflow version!)
            # Tensor Flow r0.12
            merged = tf.summary.merge_all()  # <- these work in newer versions of TF
            summary_writer = tf.summary.FileWriter('summaries/train_summary', g)
            # Tensor Flow 0.11
            # merged = tf.merge_all_summaries()
            # summary_writer = tf.train.SummaryWriter('summaries/summary', sess.graph)

            # Run training for a specific number of training examples, we will keep this in memory so we can save the
            # summary data correctly
            counter = 0

            # Run this for a specific number of intervals
            for i in range(n_epochs):

                start_time = time.time()

                # Create a session, do not output alot to the window, nor allow TF to place operations on devices of
                # their choosing
                sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

                # Trying to get queue up and running BEFORE calling the saver.restore() option which has its own init
                # function but that should be specific to the variables it saved.
                #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='initialize_op')
                sess.run(tf.local_variables_initializer())
                #init_pipeline_vars = tf.initialize_variables([images, labels])
                #sess.run(init_pipeline_vars)

                # Restore variables, also in the CIFAR-10 they use tf.train.get_checkpoint_state() to do ... which
                # seems to return the last check point
                chkp = tf.train.get_checkpoint_state('summaries/chk_pt')
                #print(chkp)
                #print(chkp.model_checkpoint_path)
                saver.restore(sess, chkp.model_checkpoint_path)
                #saver.restore(sess, 'summaries/chk_pt/model.ckpt')

                # Make a coordinator,
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                num_training_batches = int(NUM_OF_TRAINING_EXAMPLES/batch_size)
                acc_list = []

                for j in range(num_training_batches):

                    acc, summary = sess.run([accuracy, merged])
                    acc_list.append(acc)
                    summary_writer.add_summary(summary, counter) #Conter
                    counter += 1


                # Now I have to clean up
                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=4)
                sess.close()

                # Test over last batch!
                # summary, acc, predi = sess.run([merged, accuracy, key])
                print("Epoch ", i)
                print('  Accuracy:', sum(acc_list) / float(len(acc_list)))
                print('  Epoch run-time:', time.time() - start_time)

                print("\nSleeping for ", delay_time, " (secs)")
                time.sleep(delay_time)

            summary_writer.close()


if __name__ == '__main__':

    # My run-time here is about 2.6 seconds
    # My run-time on train is about 10.6 seconds
    # So evalue should rest for about 8 seconds or so

    batch_size = 500
    n_classes = 10
    n_epochs = 150
    delay_time = 2
    run_training(FILE_NAMES, batch_size, n_classes, delay_time, n_epochs)