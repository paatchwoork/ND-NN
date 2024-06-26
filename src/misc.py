from numpy import array
from numpy import uint8
from numpy import exp

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def init_first():
    return [array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
            18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
           253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
           253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
           253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
           205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
            90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
           190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
           253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
           241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
           148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
           253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
           253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
           195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
            11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=uint8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=uint8)]


# To import the whole dataset
#input_path = './input'
#training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
#training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
#test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
#test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
#
#mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
#(X_train, y_train), (X_test, y_test) = mnist_dataloader.load_data()

def S (x, l = 1):
    return (1.0 / exp(1.0 + exp(-l*x)))

def dS (x, l = 1):
    #return l*x*(1-x)
    #return (l*exp(-l*x))/((1+exp(-l*x))**2)
    return S(x)*(1-S(x))
