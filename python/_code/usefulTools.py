#!/usr/bin/env python3

"""

"""
from random import randint
import matplotlib.pyplot as plt

import sys
import os
import re
import numpy as np
from scipy import misc


def load_images(directory=".", regexp="^.*\.png$"):
    """
    Load a set of image files from a directory and specified by a regular expression
    :param directory:
    :param regexp:
    :return:
    """
    patt_re = re.compile(regexp)
    data = None
    dirlist = [f for f in os.listdir(directory) if patt_re.match(f)]
    for idx, f in enumerate(dirlist):
        print("\t\t\r  [{}] loading images {:>6}/{:}".format(f, idx+1, len(dirlist)), end="")
        sys.stdout.flush()
        if type(data) != np.ndarray:  # for the first image
            # Need to do this at first, because we do not know the image before
            img_tmp = misc.imread(directory+os.sep+f).astype(np.float)
            # prepare tensor according to image number and dimension
            data = np.ndarray([len(dirlist), 1, *img_tmp.shape], dtype=np.float16)
            data[idx, 0, :, :] = img_tmp
        else:                              # add other images
            data[idx, 0, :, :] = misc.imread(directory+os.sep+f)

    data /= 255.
    # np.savez(filename, data[type_], data_gt[type_])
    print("")

    return data


def load_dataset(display_examples=False):
    """
    Load the CROHME dataset for isolated symbols. Directories "train", "validation" and "test" must exist*
    """
    # data will store all images split in train/val/test  junk are ignored by default
    data = {i: None for i in ["train", "validation", "test"]}  # , "junk"]}  #similar to CROHME folder names
    data_gt = {i: None for i in data.keys()}
    #            data file (.nz), images
    # foreach part you set path file/ path image folder/ path to GT file
    search = {i: [False, False, False] for i in data.keys()}

    listing = [type_ for type_ in os.listdir(".")]
    for dataset in data.keys():
        for el in listing:
            if dataset in el.lower():
                if dataset == "train" and "junk" in el.lower():
                    continue
                if os.path.isfile(el):
                    if el.split(".")[-1] == "npz":
                        search[dataset][0] = el
                    elif "_GT" in el:
                        search[dataset][2] = el
                elif os.path.isdir(el):
                    search[dataset][1] = el

    patt_re = re.compile("\W*(\d*).png")

    for type_, path in search.items():
        print(" ", type_, ": ", path, sep="")

        if path[0]:  # if npz file exists, we can load it
            dt = np.load(path[0])
            data[type_] = dt["arr_0"]
            data_gt[type_] = dt["arr_1"]
            del dt
        else:                 # else load images and GT
            dirlist = [type_ for type_ in os.listdir(path[1]) if type_.split(".")[-1] == "png"]
            filename = path[1].split('_')[-1]

            data_gt[type_] = np.ndarray(len(dirlist), dtype="|S12")

            if type_ != "junk":
                with open(path[2], "r") as f:
                    gt_s = [gt.split(",")[1] for gt in f.read().split("\n") if len(gt) > 0]
            else:
                data_gt[type_][:] = "junk"

            for idx, img in enumerate(dirlist):
                img_idx = int(patt_re.search(img).group(1))

                print("\t\t\r  [{}] loading images {:>6}/{:}".format(filename, idx+1, len(dirlist)), end="")
                sys.stdout.flush()
                if type(data[type_]) != np.ndarray:  # for the first image
                    # Need to do this at first, because we do not know the image before
                    img_tmp = misc.imread(path[1]+os.sep+img).astype(np.float)
                    # prepare tensor according to image number and dimension
                    data[type_] = np.ndarray([len(dirlist), 1, *img_tmp.shape], dtype=np.float16)
                    data[type_][idx, 0, :, :] = img_tmp
                else:                              # add other images
                    data[type_][idx, 0, :, :] = misc.imread(path[1]+os.sep+img)

                if type_ != "junk":
                    data_gt[type_][idx] = gt_s[img_idx]

            data[type_] /= 255.
            np.savez(filename, data[type_], data_gt[type_])
            print("")

        if display_examples:  # for debug or fun, show some images
            for ii in range(9):
                plt.subplot(3, 3, ii+1)

                ii = randint(0, data[type_].shape[0])

                plt.title(str(ii) + " - \"" + data_gt[type_][ii].decode() + "\"")
                plt.imshow(data[type_][ii, :])
            plt.tight_layout()
            plt.show()

    return data["train"], data_gt["train"], \
        data["validation"], data_gt["validation"],\
        data["test"], data_gt["test"]


def load_mnist_dataset():
    """
    # Download and prepare the MNIST dataset
    This is just some way of getting the MNIST dataset from an online location and loading it into numpy arrays.
    It doesn't involve Lasagne at all.
    :return:
    """
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
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
    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_dataset_sequence(size):
    """
    load the dasets and reshape it using sequences instead of 2D images
    :param size:
    :return:
    """
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()
    return x_train.reshape(-1, 1, size * size), y_train, x_val.reshape(-1,  1, size * size), y_val, x_test.reshape(-1,  1, size * size), y_test


def load_dataset_sequence2(size, proj='V'):
    """
    load the data sets and project the pixel V or H to obtain sequences instead of 2D images
    :param size:
    :param proj:
    :return:
    """
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()
    x_train = x_train.reshape(-1, size, size)
    x_val = x_val.reshape(-1, size, size)
    x_test = x_test.reshape(-1, size, size)
    if proj == 'V':
        x_train = np.sum(x_train, axis=2, keepdims=True)
        x_val = np.sum(x_val, axis=2, keepdims=True)
        x_test = np.sum(x_test, axis=2, keepdims=True)
    else:
        x_train = np.sum(x_train, axis=1, keepdims=True)
        x_val = np.sum(x_val, axis=1, keepdims=True)
        x_test = np.sum(x_test, axis=1, keepdims=True)
        x_train = x_train.reshape(-1, size, 1)
        x_val = x_val.reshape(-1, size, 1)
        x_test = x_test.reshape(-1, size, 1)
    print("X shape : {}".format(x_train.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    # Batch iterator
    This is just a simple helper function iterating over training data in mini-batches of a particular size,
    optionally in random order. It assumes data is available as numpy arrays. For big datasets, you could load numpy
    arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your own custom data iteration function.
    For small datasets, you can also copy them to GPU at once for slightly improved performance. This would involve
    several changes in the main program, though, and is not demonstrated here.
    Notice that this function returns only mini-batches of size `batchsize`.
    If the size of the data is not a multiple of `batchsize`, it will not return the last (remaining) mini-batch.
    :param inputs:
    :param targets:
    :param batchsize:
    :param shuffle:
    :return:
    """
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


def evaluate(eval_fn, x, y):
    err = 0
    acc = 0
    batches = 0
    for batch in iterate_minibatches(x, y, min(y.shape[0], 50), shuffle=False):
        inputs, targets = batch
        _err, _acc = eval_fn(inputs, targets)
        err += _err
        acc += _acc
        batches += 1
    return err / batches, acc / batches


def evaluate_ae(eval_fn, x):
    err = 0
    batches = 0
    for batch in iterate_minibatches(x, x, 50, shuffle=False):
        inputs, targets = batch
        _err = eval_fn(inputs, targets)
        err += _err
        batches += 1
    return err / batches


if __name__ == "__main__":
    load_dataset()
