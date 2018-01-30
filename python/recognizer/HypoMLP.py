#!/usr/bin/env python3

from __future__ import print_function

import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as tensor
import re
import lasagne
from scipy import misc


# Config imports
# mpl.use('Agg')  # allows plotting in image without X-server

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
    datasets = ['test']
    # Deal with paths
    if paths is None:
        paths = search_paths(datasets, '.')

    # Actually loading dataset
    patt_re = re.compile('\W*(\d*).png')  # Images file name (isoX.png)
    data = {i: None for i in datasets}
    data_gt = {i: None for i in datasets}
    for corpus in datasets:
        path = paths[corpus]
        print('\t{}: {}'.format(corpus, path))
        npz_file = path[0]
        img_dir = path[1]

        if npz_file is not None and os.path.isfile(npz_file):
            # if npz file exists, we can load it
            # the npz file is created with `np.savez`
            dt = np.load(npz_file)
            data[corpus] = dt['arr_0']
            del dt
        else:
            # else load images and GT
            img_files = [file_name for file_name in os.listdir(img_dir) if file_name.split('.')[-1] == 'png']
            number_img = len(img_files)

            # Inititialize data_gt to receive ground truth data

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

            # print('\t\t\r  [{}] loading images {:>6}/{:}'.format(img, idx + 1, number_img), end='')
            sys.stdout.flush()
            print()  # Newline for logging purpose

            # Every pixel is between 0 and 255, this contains the value from 0. to 1.
            data[corpus] /= 255.
            np.savez(img_dir + '.npz', data[corpus])
    return (data['test'])

def build_mlp(input_var=None, nodes=10, size=28, label_nb=10):
    nonlin = lasagne.nonlinearities.rectify
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, size, size),
                                        input_var=input_var)

    # normal
    network = lasagne.layers.DenseLayer(network, nodes, nonlinearity=nonlin)
    # network = lasagne.layers.DenseLayer(network, nodes, nonlinearity=nonlin)
    # network = lasagne.layers.DenseLayer(network, nodes//4, nonlinearity=nonlin)
    # with drop-out
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), nodes, nonlinearity=nonlin)

    # Output layer:
    softmax = lasagne.nonlinearities.softmax

    network = lasagne.layers.DenseLayer(network, label_nb, nonlinearity=softmax)
    return network

symbols_dict = {i: idx for idx, i in enumerate(
    ["junk", "!", "(", ")", "+", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "A", "B", "C",
     "COMMA", "E", "F", "G", "H", "I", "L", "M", "N", "P", "R", "S", "T", "V", "X", "Y", "[", "\\Delta", "\\alpha",
     "\\beta", "\\cos", "\\div", "\\exists", "\\forall", "\\gamma", "\\geq", "\\gt", "\\in", "\\infty", "\\int",
     "\\lambda", "\\ldots", "\\leq", "\\lim", "\\log", "\\lt", "\\mu", "\\neq", "\\phi", "\\pi", "\\pm", "\\prime",
     "\\rightarrow", "\\sigma", "\\sin", "\\sqrt", "\\sum", "\\tan", "\\theta", "\\times", "\\{", "\\}", "]", "a", "b",
     "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
     "z", "|"])}


def main():

    # Load the dataset
    print("Loading data...")
    base = 'LGs'
    test_base = 'hypo_png'
    x_test = load_dataset(
        paths={
            'test': [os.path.join(base, 'testSymbols.npz'),
                     os.path.join(base, 'hypo_png')]})
    # Prepare Theano variables for inputs and targets
    input_var = tensor.tensor4('inputs')
    target_var = tensor.ivector('targets')
   
    network = build_mlp(input_var, nodes=150, size=x_test.shape[-1], label_nb=len(symbols_dict))

    # Create neural network model (depending on first command line parameter)
    print("load model")
    with np.load('modelMLP.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    prediction = lasagne.layers.get_output(network, deterministic=True)

    pred_fn = theano.function([input_var], prediction) 
    ma_prediction = pred_fn(x_test)

if __name__ == '__main__':
    main()
