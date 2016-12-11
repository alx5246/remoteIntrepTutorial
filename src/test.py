# This is the test file as in
# https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d#.ox1kxnl24

import tensorflow as tf
import matplotlib

print("Tensorflow Imported !!!")

matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.arange(100))
plt.show()