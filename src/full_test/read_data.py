# A. lons
# Decemebr 2016
#
# This is part of the full test suite, here we are responsible for loading and handling the data! In particular I used
# much from other repositories; links to said repositories are as follows,
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
#


import tensorflow as tf
from tensorflow.python.client import timeline

def read_binary_image(filename_queue):
    """
    DESCRIPTION
    This is originally taken from teh cifar-10 example (cifar10_input.read_cifar10()), and then modified for my
    purposes. It should work in a similar way.
    NOTE
    From the original file "Reads and parses examples from CIFAR10 data files."
    ARGS:
        a filename_queue: A queue of strings with the filenames to read from.
    RETURNS:
        An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
        for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    with tf.name_scope('binary_image_reader'):

        # Using the with gpu fails! These things there cannot be put on GPU!
        #with tf.device('/cpu:0'):
        with tf.device('/cpu:0'):

            # (AJL) make a dummy class? They do this in the examples...
            class CIFAR10Record(object):
                pass
            result = CIFAR10Record()

            # Dimensions of the images in the CIFAR-10 dataset, input format. The following are done to find the size of
            # the files!
            label_bytes = 1  # 2 for CIFAR-100
            result.height = 32
            result.width = 32
            result.depth = 3
            image_bytes = result.height * result.width * result.depth
            # Every record consists of a label followed by the image, with a fixed number of bytes for each.
            record_bytes = label_bytes + image_bytes

            # Read a record, getting file names from the filename_queue. Readers are tensorflows way for reading data formats.
            # No header or footer in the CIFAR-10 format, so we leave header_bytes and footer_bytes at their default of 0.
            reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, name='record_reader')
            # (AJL) the "reader" will dequeue a work unit from teh queue if necessary (e.g. when the Reader needs to start
            # reading from a new file since it has finished with the previous file)
            result.key, value = reader.read(filename_queue, name='reading_record')

            # Convert from a string to a vector of uint8 that is "record_bytes" long, that is we
            example_in_bytes = tf.decode_raw(value, tf.uint8, name='decoding_raw_bytes')

            # The first bytes represent the label, which we convert from uint8->int32. This is just the label
            result.label = tf.cast(tf.slice(example_in_bytes, [0], [label_bytes], name='slice_label_bytes'), tf.int32, name='cast_label_bytes')
            # We make this scalar label a one-hot type
            result.label = tf.one_hot(result.label, depth=10, name='label_scaler_to_vec')
            # Flatten label to 1D, rather than 2D with 1-row
            result.label = tf.reshape(result.label, [-1], name='label_flatten')

            # The remaining bytes after the label represent the image, which we reshape from [depth * height * width] to [depth,
            # height, width]. I change from tf.strided_slice(), because it was not working, and instead went to tf.slice so I
            # also had to change the paramters as they are different for the different function. Thus here I am returning
            # a single sliced imaged.
            ourImage = tf.slice(example_in_bytes, [label_bytes], [image_bytes], name='slice_image_bytes')

            # Reshape the image into its proper form
            depth_major = tf.reshape(ourImage, [result.depth, result.height, result.width], name='image_first_reshaping')

            # Convert from [depth, height, width] to [height, width, depth].
            result.uint8image = tf.transpose(depth_major, [1, 2, 0], name='image_transposing')

            # Casting, I am not sure if I should be doing this?
            result.fl32_image = tf.cast(result.uint8image, tf.float32, name='cast_image_to_fl32')

    return result


def input_pipline(file_names, batch_size, numb_pre_threads, num_epochs = 1):
    """
    DESCRIPTION
        In accordance with your typical pipeline, we have a seperate method that sets up the data.
    :param file_names: list of file names that have the data
    :param batch_size: the number of examples per batch
    :param numb_pre_threads:
    :return: A tuple (images, labels, keys) where:
    """

    # This will no work if we pin to the GPU!
    with tf.device('/cpu:0'):

        with tf.name_scope('input_pipeline'):

            # Generate the file-name queue from given list of filenames. IMPORTANT, this function can read through strings
            # indefinitely, thus you WANT to give a "num_epochs" parameter, when you reach the limit, the "OutOfRange" error
            # will be thrown.
            filename_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, name='file_name_queue')

            # Read the image using method defined above, this will actually take the queue and one its files, and read some data
            read_input = read_binary_image(filename_queue)

            # Use tf.train.shuffle_batch to shuffle up batches. "min_after_dequeue" defines how big a buffer we will
            # randomly sample from -- bigger means better shuffling but slower start up and more memory used. "capacity"
            # must be larger than "min_after_dequeue" and the amount larger determines the maximm we will prefetch. The
            # recommendation: for capacity is min_after_dequeue + (num_threads + saftey factor) * batch_size
            # From cifar10_input.input(), setup min numb of examples in the queue
            min_fraction_of_examples_in_queue = .6
            min_queue_examples = int(batch_size * min_fraction_of_examples_in_queue)
            min_after_dequeue = min_queue_examples
            capacity = min_queue_examples + 3 * batch_size

            # If I want to shuffle input!
            """
            images, label_batch, key = tf.train.shuffle_batch([read_input.fl32_image, read_input.label, read_input.key],
                                                              batch_size=batch_size, num_threads=numb_pre_threads,
                                                              capacity=capacity, min_after_dequeue=min_after_dequeue,
                                                              name='train_shuffle_batch')
            """
            # If I do not wany to shuffle input!

            images, label_batch, key = tf.train.batch([read_input.fl32_image, read_input.label, read_input.key],
                                                      batch_size=batch_size, num_threads=numb_pre_threads,
                                                      capacity=capacity,
                                                      name='batch_generator')


        return images, label_batch, tf.reshape(key, [batch_size])


if __name__ == '__main__':

    #Here we will run the test! This will test our abilities to set everything correctly!

    # Get file names by setting up my readers and queues,
    #  see, (https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py)
    # in method, inputs(), I think I can "Force all teh input processing onto the CPU" by calling the tf.device here as
    # long as I have "allow_soft_placement=False" in the session settings.
    with tf.device('/cpu:0'):
        filenames = ['cifar-10-batches-bin/data_batch_1.bin']
        images, labels, key = input_pipline(filenames, batch_size=10, numb_pre_threads=8)


    # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the tf.local_variables_init()
    # because I set the num_epoch in the string producer in the other python file.
    # Tensor Flow r0.12
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # In tf 011
    #init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables(), name='initialize_ops')


    # Create a session, this is done in how-to and cifar-10 example (in the cifar-10 the also have some configs). I use
    # "log_device_placement" because this will output a bunch of stuff.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=False))

    # Now prepare all summaries (these following lines will be be based on the tensorflow version!)
    # Tensor Flow r0.12
    mergedSummaries = tf.summary.merge_all()  # <- these work in newer versions of TF
    summary_writer = tf.summary.FileWriter('summaries/train_summary', sess.graph)
    # Tensor Flow 0.11
    #merged = tf.merge_all_summaries()
    #summary_writer = tf.train.SummaryWriter('summaries/summary', sess.graph)

    # I need to run meta-data
    #   a) To get a time-line to work see running meta-data see http://stackoverflow.com/questions/40190510/tensorflow-
    #   how-to-log-gpu-memory-vram-utilization/40197094
    #   b) To get detailed run information to text file see http://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # Run the init, this is done in how-to and cifar-10
    sess.run(init_op)

    # Make a coordinator,
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for i in range(10):

        a, b, c = sess.run([images, labels, key], options=run_options, run_metadata=run_metadata)

        print("\n")
        print("IMAGE: of type %s and size %s" % (type(a), a.shape))
        print("IMAGE: is of ")
        print("LABELS: of type %s and size %s" % (type(b), b.shape))

        # with open("meta_data_run.txt", "w") as out:
        #    out.write(str(run_metadata))
        t1 = timeline.Timeline(run_metadata.step_stats)
        ctf = t1.generate_chrome_trace_format()
        with open('summaries/timelines/timeline.json', 'w') as f:
            f.write(ctf)

    # Close this up
    summary_writer.close()


    # Now I have to clean up
    coord.request_stop()
    coord.join(threads)
    sess.close()