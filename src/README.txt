A. Lons
December 2016

README for all files in src/

These are tensorflow tutorials, which are all run over remote and on GPU. To start off I had to figure out how to config.
the IDE to do this (see A&S robotics in setup).

########################################################################################################################
Files/Folders here,

A) cifar10_examples/
    This folder has some downloaded/copied examples from the tensorflow repository and are here for me to learn from.

B) sentdex_tutorials/
    All files based on sentdex's youtube.com video series "Machine Learning with Python". In particular, these files are
    those from his vidoes on using GPU computing.


########################################################################################################################
GPU USEAGE, GPU MEMORY PLACEMENT, PAPERS AND INSIGHT

1) A multi-GPU example,
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/multiple_gpu.py

2) The CIFAR10 has a good multi-gpu example with using cpu and gpu in conjunction! It seems that it first pins everying
   to the cpu in the outer section of the run, and then specifically calls the training part to the gpu.
   https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

3) Running our of memmory with MNIST data? I was but according to this post, it might not be actually from training
   but instead from the testing part. In the example (from tf) they test with the whole dataset, which is perhaps,
   too big. It turns out for MNIST at least this worked for me!
   http://cpaxton.github.io/2016/05/11/cuda-tensorflow/

4) Best allocation of GPU vs CPU
    http://www.itgo.me/a/5542834503577774338/tensorflow-multi-gpu-configuration-performance

5) Where to pin memory on CPU or GPU, at the middle and bottom of post they talk abotu where to best to pin things to
    https://github.com/tensorflow/tensorflow/issues/838

6) It turns out, if a GPU is found, TF will try to PLACE EVERYTHING it can on teh GPU, however some operations do not
    have a GPU implementation and this greedy placement technqiue can be suboptimal due to data transfers. One way to
    find out if an op has GPU implement is to search for "REGISTER_KERNEL _BUILDER" macros, (see end of the first and
    second posts below). It is best to "pin the whole input pipeline to cpu manually"
    https://github.com/tensorflow/tensorflow/issues/838
    https://github.com/tensorflow/tensorflow/issues/975

7) Adam optimizer can thriple memory use, becasue it looks at means, and std of all variables!
   http://stackoverflow.com/questions/36390767/tensor-flow-ran-out-of-memory-trying-to-allocate

8) A good thing that tells us how to handle or setup configuration files. In particular there are many options like
    "allow_soft_placement" or "device_count" ... etc. The second of the references below, uses things like meta-data and
    more outputs.
    https://github.com/tensorflow/tensorflow/issues/838
    META-DATA: http://stackoverflow.com/questions/40190510/tensorflow-how-to-log-gpu-memory-vram-utilization/40197094

9) In fully_connected_preloaded.py, it seems that the they only put the inputs in the with tf.device('/cpu') block. The
    rest of the operations seem to be outside of that. I am still not sure how to best apply the tf.device() command or
    where to put it.
    see, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_preloaded.py

10) Talking about how the tf.device() pinning may work, and that need to you need set allow_soft_placement=True, so the
    run time does not override your settings.

1) For question essentailly that boils down to passing data form input pipeline to gpu/gpus. This is a good example,
   because it talks about allocating to and from the GPUs and how to call them and CPUs.
   "FIFOQueue: dequeue many operation very slow"
   https://github.com/tensorflow/tensorflow/issues/3009

12) This Tutorial is stricktly here to see if I can get up and running by using an interpreture from another machine!
    see, https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d#.ox1kxnl24

13) Pin the whole input-pipeline to the CPU, this is Yaroslav's constant notes, which he notes before but
    see, http://stackoverflow.com/questions/36950299/tensorflow-0-6-gpu-issue/36950646#36950646. In particular the in
    another things the Yaroslav pins, the tf.constants to the cpu see, ...
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_preloaded.py

14) In this post, mmry says "You would use a 'with tf.device("/cpu:0")' block to wrap the construction of the ops in
    the input pipeline', see,
    http://stackoverflow.com/questions/36313934/is-it-possible-to-split-a-network-across-multiple-gpus-in-tensorflow

15) I found this tricks of the trade (video) that explains a bunch about assignign to devices!
    TensorFlow: Tricks of the Trade (Video and Slides) by Yaroslav Bulatov
    https://www.meetup.com/TensorFlow-London/events/229662676/ (this is where the video recorded)
    https://blog.altoros.com/video-and-slides-from-the-tensorflow-london-meetup-march-31-2016.html?utm_content=buffer09eba&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer&utm_source=youtube_channel&utm_medium=organic&utm_campaign=youtube
        -Notes
        - Queues: Usually driven from threads, generally FIFO queue
        - Assign Devices to Ops when building the graph, if you go between CPU and GPU, under the hood TF adds 'Send'
        and 'recieve' nodes for copying of data. The video has pretty graph ~14:00, on how things get set.
        - You can assign things manually, which is what we need to do. Current TF, just will place everyting on GPU if
        you can.

16) A tutorial about using GPUs and importantly WHAT TO OPS TO SEND TO GPU! The author notes any large matrix operations
    should be sent to the GPU as a rule of thumb. One way to make sure that ops not allowed on GPU is to set the
    allow_soft_placement=True option in the tf.Session(config=tf.ConfigProto(allow_soft_placement=True)), but I am not
    sure I want to give us this control...., another thing to do is set up the config to log_device_placement=True.
    see, http://learningtensorflow.com/lesson10/

17) How to setup a loop so handle runnning out of data, see fully_connected_reader.py

18) To see yaroslavvb's how to generate text file you should run meta data
    https://github.com/yaroslavvb/stuff/commit/aa886026ca3c48e27f776269548b40a2e2bb89ea

19) To see how to appropriately use the 'timelime', which outputs a very nice gprahical info,
    see http://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow

20) Handling running out of examples, the fully_connected_ready.py example shows how to setup a loop to handle running
    out of examples!
    see, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py





