A. Lons
December 2016

README for all files in src/full_test

Here I want to play with a full test on the GPU, specifically with proper allocation of things pinned to GPU vs. CPU.
Here in the code (I will document the results here) I will play with trying different placements to see how it effects
the run times.

########################################################################################################################
References that helped here,

1) Apparently the best way to train/test is to create seperate graphs for each.
    http://stackoverflow.com/questions/37801137/duplicate-a-tensorflow-graph

2) How to save and load variables for inference graph,
    http://stackoverflow.com/questions/34454901/transfer-parameters-from-training-to-inference-graph

3) Stuggling to get the eval() graph to work because the queue is empty... however it sounds like this is because I
    should only be restoring trainable variables
    http://stackoverflow.com/questions/37632102/tensorflow-trouble-re-opening-queues-after-restoring-a-session

########################################################################################################################
What have I learned here?

1) Queues and readers cannot be pinned to the GPU
    In the read_data.py (if I run this alone), I added some pin to GPU, in particular input_pipeline(). When I do this
    it automatically throws errors because, according to the output , QueueEnqueue, QueueSize, QueueClose, and
    RandomShuffleQueue cannot run on GPU! Thus in many examples I have seen online where they apparently pin to the GPU
    but then it runs alright must be because they are running allow_soft_placement=True which I think is probably the
    default option.
    To be clear, given allow_soft_placement=False
        a) main input_pipeline pinned to CPU, input_pipline() pinned to CPU, read_binary_image() pinned to CPU
            RUNS
        b) main input_pipeline pinned to CPU, input_pipline() pinned to CPU, read_binary_image() pinned to GPU
            FAILS
        c) main input_pipeline pinned to CPU, input_pipline() pinned to GPU, read_binary_image() pinned to CPU
            FAILS
        d) main input_pipeline pinned to GPU, input_pipline() pinned to CPU, read_binary_image() pinned to CPU
            RUNS
    given allow_soft_placement=True
        e) main input_pipeline pinned to GPU, input_pipline() unpinned, read_binary_image() unpinned
            RUNS

2) Finally got the evaluate thing to work by first calling an initializer() op, then restoring a saved checkpoint, but
    using a special call: saver = tf.train.Saver(tf.trainable_variables())
