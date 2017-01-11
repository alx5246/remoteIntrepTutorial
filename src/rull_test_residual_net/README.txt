A. Lons
Jan 2016

README for all files in src/full_test_residual_net

To be specific, all code is written for a single GPU (though evaluate and train will be run on different GPUs)

Here I want to play with a full test on the GPU, specifically with proper allocation of
a) how to implement a residual network in tensor flow given variables will hae to be shared and what not.

Here in the code (I will document the results here) I will play with trying different placements to see how it effects
the run times.

########################################################################################################################
References that helped here with my codeing

1) A simple and concise TF Residual Network,
    see, https://github.com/xuyuwei/resnet-tf