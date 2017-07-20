import gzip
import pickle


def data_init(file_path, shape='vector', mode='train'):
    assert shape == 'vector' or 'image'
    assert mode == 'train' or 'test'
    print("Loading MNIST ...    ", end="")
    f = gzip.open(file_path, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    if shape == 'image':
        train_x = train_set[0].reshape((train_set[0].shape[0], 28, 28, 1))
        valid_x = valid_set[0].reshape((valid_set[0].shape[0], 28, 28, 1))
        test_x = test_set[0].reshape((test_set[0].shape[0], 28, 28, 1))
    else:
        train_x = train_set[0]
        valid_x = valid_set[0]
        test_x = test_set[0]
    print("COMPLETE")
    if mode == 'train':
        return train_x, valid_x
    else:
        return test_x
