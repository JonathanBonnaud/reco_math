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

import lasagne

import code1.usefulTools as Tools

# Config imports
mpl.use('Agg')  # allows plotting in image without X-server


# ##################### Build the neural network model #######################


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


# ############################## Main program ################################

symbols_dict = {i: idx for idx, i in enumerate(
    [" junk", "!", "(", ")", "+", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "A", "B", "C",
     "COMMA", "E", "F", "G", "H", "I", "L", "M", "N", "P", "R", "S", "tensor", "V", "X", "Y", "[", "\\Delta", "\\alpha",
     "\\beta", "\\cos", "\\div", "\\exists", "\\forall", "\\gamma", "\\geq", "\\gt", "\\in", "\\infty", "\\int",
     "\\lambda", "\\ldots", "\\leq", "\\lim", "\\log", "\\lt", "\\mu", "\\neq", "\\phi", "\\pi", "\\pm", "\\prime",
     "\\rightarrow", "\\sigma", "\\sin", "\\sqrt", "\\sum", "\\tan", "\\theta", "\\times", "\\{", "\\}", "]", "a", "b",
     "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
     "z", "|"])}


def char2intvec(vect):
    for i in range(vect.shape[0]):
        vect[i] = symbols_dict[vect[i].decode()]
    return vect.astype(np.uint8)


def main():
    num_epochs = 10

    # Load the dataset
    print("Loading data...")
    x_train, y_train, x_val, y_val, x_test, y_test = Tools.load_mnist_dataset()

    # Turn char label vector into uint8 vector , NOT needed with MNIST dataset
    # y_train = char2intVec(y_train)
    # y_val = char2intVec(y_val)
    # y_test = char2intVec(y_test)

    # Get index value of junk symbol in symbol list
    # junk_idx = symbols_dict[" junk"]
    # Get boolean arrays containing false where junk symbols appear
    # idx_junk_train = y_train != junk_idx
    # idx_junk_val = y_val != junk_idx
    # idx_junk_test = y_test != junk_idx

    # Remove junk symbols from dataset - selecting data where boolean arrray is True
    # y_train = y_train[idx_junk_train]; x_train = x_train[idx_junk_train]
    # y_val = y_val[idx_junk_val]; 	     x_val = x_val[idx_junk_val]
    # y_test = y_test[idx_junk_test];    x_test = x_test[idx_junk_test]

    # Prepare Theano variables for inputs and targets
    input_var = tensor.tensor4('inputs')
    target_var = tensor.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_mlp(input_var, nodes=150, size=x_train.shape[-1], label_nb=len(symbols_dict))

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):

    prediction = lasagne.layers.get_output(network, deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # As a bonus, also create an expression for the classification accuracy:
    # prediction = lasagne.layers.get_output(network)
    test_acc = tensor.mean(tensor.eq(tensor.argmax(prediction, axis=1), target_var),
                           dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [loss, test_acc])

    # save the results
    val_err_array = np.array([])
    train_err_array = np.array([])
    best_val_mse = 1000000
    best_epoch = 0

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in Tools.iterate_minibatches(x_train, y_train, 50, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        train_err = train_err / train_batches

        # And a full pass over the validation data:
        train_err, train_acc = Tools.evaluate(val_fn, x_train, y_train)
        val_err, val_acc = Tools.evaluate(val_fn, x_val, y_val)
        val_err_array = np.append(val_err_array, val_err)
        train_err_array = np.append(train_err_array, train_err)

        # instead of saving for each epoch, save only when MSE is better
        # np.savez('modelMLP.npz', *lasagne.layers.get_all_param_values(network))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        print("  train accuracy:\t\t{:.2f} %".format(
            train_acc * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc * 100))

    # with np.load('modelMLP.npz') as f:
    # 	 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    # After training, we compute and print the test error:
    test_err, test_acc = Tools.evaluate(val_fn, x_test, y_test)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc * 100))

    print(val_err_array)
    plt.xlabel('epoch')
    plt.ylabel('MSE val / train')
    plt.title('Digit classifier')
    plt.plot(range(num_epochs), val_err_array, 'b', range(num_epochs), train_err_array, 'r', [best_epoch],
             [best_val_mse], 'go')
    plt.savefig('resultMLP.png')


if __name__ == '__main__':
    main()
