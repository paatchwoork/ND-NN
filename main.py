from src.NNClasses import Network

from src.MnistDataloader import MnistDataloader

from src.misc import init_first
from src.misc import flatten

import numpy as np
from os.path  import join

pic_size = 28

layout = [pic_size**2,
        16,
        16,
        10]

nw = Network(layout)

input_path = './input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(X_test, y_test) = mnist_dataloader.load_test()

nw.read(X_test[0])
label = y_test[0]


nw.propagate()
tot_err = nw.calculate_errors(label)
print(f'Total error = {tot_err}')
nw.backprop()
