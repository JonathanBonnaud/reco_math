#!/usr/bin/env python3
import math
import os
import re
import sys
from random import randint

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


# Load a set of image files from a directory and specified by a regular expression
def load_images(directory='.', regexp='^.*\.png$'):
    patt_re = re.compile(regexp)
    data = None
    dirlist = [f for f in os.listdir(directory) if patt_re.match(f)]
    for idx, f in enumerate(dirlist):
        print('\t\t\r  [{}] loading images {:>6}/{:}'.format(f, idx + 1, len(dirlist)), end='')
        sys.stdout.flush()
        if type(data) != np.ndarray:  # for the first image
            # Need to do this at first, because we do not know the image before
            img_tmp = misc.imread(directory + os.sep + f).astype(np.float)
            # prepare tensor according to image number and dimension
            data = np.ndarray([len(dirlist), 1, *img_tmp.shape], dtype=np.float16)
            data[idx, 0, :, :] = img_tmp
        else:  # add other images
            data[idx, 0, :, :] = misc.imread(directory + os.sep + f)

    data /= 255.
    # np.savez(filename, data[type_], data_gt[type_])
    return data


def search_paths(datasets, root):
    # foreach corpus we search for [npz file, image folder, GT file]
    # if no paths were given then we try to find the paths in '.'
    search = {i: [None, None, None] for i in datasets}

    files = [file_name for file_name in os.listdir(root)]
    for corpus in datasets:
        for file_name in files:
            if corpus in file_name.lower():
                if corpus == 'train' and 'junk' in file_name.lower():
                    continue
                if os.path.isfile(file_name):
                    if file_name.split('.')[-1] == 'npz':
                        search[corpus][0] = file_name
                    elif '_GT' in file_name:
                        search[corpus][2] = file_name
                elif os.path.isdir(file_name):
                    search[corpus][1] = file_name
    return search


def display_img(data, labels):
    nb_img = len(data)
    squared = math.ceil(math.sqrt(nb_img))
    for ii in range(nb_img):
        plt.subplot(squared, squared, ii + 1)

        plt.title(str(ii) + ' - \'' + labels[ii].decode() + '\'')
        plt.imshow(data[ii])
    plt.tight_layout()
    plt.show()


def load_dataset(paths=None, display_examples=False, load_junk=False):
    """
        Load CROHME's isolated symbols datasets
        Loading a dataset will create a new file .npz that is binary and REALLY
        accelerate the loading of the data
        If load_junk is true then data['train'] will contain train symbols
        and junk symbols at the end

        paths is a dict : {corpus_name: [path_to_npz_file, path_to_img_dir, path_to_gt_file]}
        You should pass this argument else the program will try to find itself the paths

        This function assert that the the GT file and the img_files are in the same order
        the first line of GT file concerns the img file with the lowest number
        this can be dangerous /!\

        Note: GT stand for Ground Truth
    """
    datasets = ['train', 'validation', 'test', 'mydata']
    if load_junk:
        datasets += ['junk']

    # Deal with paths
    if paths is None:
        paths = search_paths(datasets, '.')

    # Actually loading dataset
    patt_re = re.compile('\W*(\d*).png')  # Images file name (isoX.png)
    data = {i: None for i in datasets}
    data_gt = {i: None for i in datasets}
    data_ids = []
    for corpus in datasets:
        path = paths[corpus]
        # print('\t{}: {}'.format(corpus, path))
        npz_file = path[0]
        img_dir = path[1]
        gt_file = path[2]

        if npz_file is not None and os.path.exists(npz_file):
            # if npz file exists, we can load it
            # the npz file is created with `np.savez`
            print("Loading file {}...".format(npz_file))
            dt = np.load(npz_file)
            data[corpus] = dt['arr_0']
            data_gt[corpus] = dt['arr_1']
            del dt
        else:
            # else load images and GT
            img_files = [file_name for file_name in os.listdir(img_dir) if file_name.split('.')[-1] == 'png']
            number_img = len(img_files)

            # Inititialize data_gt to receive ground truth data
            data_gt[corpus] = np.ndarray(number_img, dtype='|S12')  # '|S12' stands for 12 char string

            # If the corpus is junk then we don't need to get the GT as it is always 'junk'
            if corpus != 'junk' and corpus != 'mydata':
                # Loading GT file
                with open(gt_file, 'r') as f:
                    gt_s = [gt.split(',')[1].strip() for gt in f.read().split('\n') if len(gt) > 0]

            log_every = max(int(number_img * 0.01), 1)
            for idx, img in enumerate(img_files):
                if idx % log_every == 0:
                    # Do not log every file for performance purposes
                    print('\t\t\r  [{}] loading images {:>6}/{:}'.format(img, idx + 1, number_img), end='')
                    sys.stdout.flush()

                img_idx = int(patt_re.search(img).group(1))
                loaded_img = misc.imread(os.path.join(img_dir, img)).astype(np.float)

                # Initialize data[corpus] with the dimensions of the images
                if type(data[corpus]) != np.ndarray:  # for the first image
                    data[corpus] = np.ndarray([number_img, 1, *loaded_img.shape], dtype=np.float16)

                # Add image to data
                data[corpus][idx, 0, :, :] = loaded_img

                # Add GT for the image
                if corpus != 'junk' and corpus != 'mydata':
                    data_gt[corpus][idx] = gt_s[img_idx]
                elif corpus != 'mydata':
                    data_gt[corpus][idx] = 'junk'
                else:
                    data_ids.append(img)
                print('\t\t\r  [{}] loading images {:>6}/{:}'.format(img, idx + 1, number_img), end='')
            sys.stdout.flush()
            print()  # Newline for logging purpose

            # Every pixel is between 0 and 255, this contains the value from 0. to 1.
            data[corpus] /= 255.
            np.savez(img_dir + '.npz', data[corpus], data_gt[corpus])

        if display_examples:  # for debug or fun, show some images
            nb_img = 9
            rand_indexes = [randint(0, data[corpus].shape[0]) for _ in range(nb_img)]
            display_img([data[corpus][i, 0, :].astype(np.float32) for i in rand_indexes],
                        [data_gt[corpus][i] for i in rand_indexes])

    if load_junk:
        data['train'] = np.concatenate([data['train'], data['junk']])
        data_gt['train'] = np.concatenate([data_gt['train'], data_gt['junk']])

    return (data['train'], data_gt['train'],
            data['validation'], data_gt['validation'],
            data['test'], data_gt['test'],
            data['mydata'], data_ids)


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.
def load_MNISTdataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print('Downloading %s' % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('mnist/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('mnist/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('mnist/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('mnist/t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# load the dasets and reshape it using sequences instead of 2D images
def load_datasetSequence(size):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train = X_train.reshape(-1, 1, size * size)
    X_val = X_val.reshape(-1, 1, size * size)
    X_test = X_test.reshape(-1, 1, size * size)
    return X_train, y_train, X_val, y_val, X_test, y_test


# load the dasets and project the pixel V or H to obtain sequences instead of 2D images
def load_datasetSequence2(size, proj='V'):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train = X_train.reshape(-1, size, size)
    X_val = X_val.reshape(-1, size, size)
    X_test = X_test.reshape(-1, size, size)
    if (proj == 'V'):
        X_train = np.sum(X_train, axis=2, keepdims=True)
        X_val = np.sum(X_val, axis=2, keepdims=True)
        X_test = np.sum(X_test, axis=2, keepdims=True)
    else:
        X_train = np.sum(X_train, axis=1, keepdims=True)
        X_val = np.sum(X_val, axis=1, keepdims=True)
        X_test = np.sum(X_test, axis=1, keepdims=True)
        X_train = X_train.reshape(-1, size, 1)
        X_val = X_val.reshape(-1, size, 1)
        X_test = X_test.reshape(-1, size, 1)
    print('X shape : {}'.format(X_train.shape))
    return X_train, y_train, X_val, y_val, X_test, y_test


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def evaluate(eval_fn, X, y):
    err = 0
    acc = 0
    batches = 0
    for batch in iterate_minibatches(X, y, min(y.shape[0], 50), shuffle=False):
        inputs, targets = batch
        _err, _acc = eval_fn(inputs, targets)
        err += _err
        acc += _acc
        batches += 1
    return err / batches, acc / batches


def evaluateAE(eval_fn, X):
    err = 0
    batches = 0
    for batch in iterate_minibatches(X, X, 50, shuffle=False):
        inputs, targets = batch
        _err = eval_fn(inputs, targets)
        err += _err
        batches += 1
    return err / batches


def main():
    base = '../../data/DataSymbol_Iso/'
    my_base = '../../LGs/'
    test_base = 'task2-testSymbols2014'
    train_base = 'task2-trainSymb2014'
    validation_base = 'task2-validation-isolatedTest2013b'
    load_dataset(paths={'test': [os.path.join(base, test_base, 'img_testSymbols.npz'),
                                 os.path.join(base, test_base, 'img_testSymbols'),
                                 os.path.join(base, test_base, 'testSymbols_2016_iso_GT.txt')],
                        'validation': [os.path.join(base, validation_base, 'img_validationSymbols.npz'),
                                       os.path.join(base, validation_base, 'img_validationSymbols'),
                                       os.path.join(base, validation_base, 'validationSymbols', 'iso_GT.txt')],
                        'train': [os.path.join(base, train_base, 'img_trainingSymbols.npz'),
                                  os.path.join(base, train_base, 'img_trainingSymbols'),
                                  os.path.join(base, train_base, 'trainingSymbols', 'iso_GT.txt')],
                        'junk': [os.path.join(base, train_base, 'img_trainingJunk.npz'),
                                 os.path.join(base, train_base, 'img_trainingJunk'),
                                 os.path.join(base, train_base, 'trainingJunk', 'iso_GT.txt')],
                        'mydata': [
                            os.path.join(my_base, 'img_segments_hypo.npz'),
                            os.path.join(my_base, 'img_segments_hypo'),
                            None
                        ]},
                 display_examples=False,
                 load_junk=True)

if __name__ == '__main__':
    main()
