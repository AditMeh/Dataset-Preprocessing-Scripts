import numpy as np

FILEPATH = "cifar-10-batches-py/"
data_batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_path = "test_batch"

# reads the batch files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# transforms the (32*32*3, ) vector into a (32, 32, 3) matrix, representing the image
def morph_image(arr):
    r = arr[:1024].reshape((32, 32))
    g = arr[1024:2048].reshape((32, 32))
    b = arr[2048:3072].reshape((32, 32))
    image_arr = np.dstack((r, g, b))
    return image_arr


# input: number of batches to be used for the training set generation
# output: train_X and train_y, arrays of dimensions (10000*num_batches, 32, 32, 3) and (10000*num_batches, 10)
# respectively.
def prepare_train(num_batches):
    train_X = []
    train_y = []
    assert (num_batches <= 5)

    for i in range(num_batches):
        batch_file = unpickle(FILEPATH + data_batches[i])

        set_size = len(batch_file[b'labels'])
        for f in range(set_size):
            one_hot = [0 for z in range(10)]
            one_hot[batch_file[b'labels'][f]] = 1
            test_y.append(one_hot)

            image_arr = morph_image(batch_file[b'data'][f])
            test_X.append(image_arr)

    return np.array(train_X), np.array(train_y)


# input: none
# output: text_X and test_y, two arrays of sizes (100000,32,32, 3) and (10000,10) respectively
def prepare_test():
    test_X = []
    test_y = []

    batch_file = unpickle(FILEPATH + test_path)

    set_size = len(batch_file[b'labels'])
    for f in range(set_size):
        one_hot = [0 for z in range(10)]
        one_hot[batch_file[b'labels'][f]] = 1
        test_y.append(one_hot)

        image_arr = morph_image(batch_file[b'data'][f])
        test_X.append(image_arr)

    return np.array(test_X), np.array(test_y)


# input: batch file name
# output: array of length = 10, where each index corresponds to a collection of images belonging to the class
# at that specific index within the given batch. From observation, most classes will be evenly distributed.
def segment_image_in_batch(batch_path):
    batch_file = unpickle(FILEPATH + batch_path)
    x, y = batch_file[b'data'], batch_file[b'labels']

    sample_set = [[] for i in range(10)]

    for element_index in range(len(x)):
        element_class = y[element_index]
        sample_set[element_class].append(morph_image(x[element_index]))

    numpy_sample_set = []
    for arr in sample_set:
        numpy_sample_set.append(np.array(arr))

    return np.asarray(numpy_sample_set)
