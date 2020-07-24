import numpy as np
import imageio

FILEPATH = "cifar-10-batches-py/"
data_batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_path = "test_batch"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def prepare_train(num_batches):
    train_X = []
    train_y = []
    assert (num_batches <= 5)

    for i in range(num_batches):
        batch_file = unpickle(FILEPATH + data_batches[i])
        for f in batch_file[b'labels']:
            one_hot = [0 for z in range(10)]
            one_hot[f] = 1
            train_y.append(one_hot)
        for features in batch_file[b'data']:
            r = features[:1024].reshape((32, 32))
            g = features[1024:2048].reshape((32, 32))
            b = features[2048:3072].reshape((32, 32))
            image_arr = np.dstack((r, g, b))
            train_X.append(image_arr)
    return np.array(train_X), np.array(train_y)


def prepare_test():
    test_X = []
    test_y = []

    batch_file = unpickle(FILEPATH + test_path)

    for f in batch_file[b'labels']:
        one_hot = [0 for z in range(10)]
        one_hot[f] = 1
        test_y.append(one_hot)
    for features in batch_file[b'data']:
        r = features[:1024].reshape((32, 32))
        g = features[1024:2048].reshape((32, 32))
        b = features[2048:3072].reshape((32, 32))
        image_arr = np.dstack((r, g, b))
        test_X.append(image_arr)

    return np.array(test_X), np.array(test_y)


train_X, train_y = prepare_train(5)

print(train_X.shape)
print(train_y.shape)

test_X, test_y = prepare_test()

print(test_X.shape)
print(test_y.shape)

imageio.imwrite("bobo.jpg", test_X[400])