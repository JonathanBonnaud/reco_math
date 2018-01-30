from sklearn.preprocessing import LabelBinarizer
import numpy as np
import theano
import theano.tensor as T

import lasagne
from pathlib import Path
from python.code.usefulTools import load_images
import time


def build_cnn(input_var=None):
  # As a third model, we'll create a CNN of two convolution + pooling stages
  # and a fully-connected hidden layer in front of the output layer.

  # Input layer, as usual:
  network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                      input_var=input_var)
  # This time we do not apply input dropout, as it tends to work less well
  # for convolutional layers.

  # Convolutional layer with 32 kernels of size 5x5. Strided and padded
  # convolutions are supported as well; see the docstring.
  network = lasagne.layers.Conv2DLayer(
          network, num_filters=32, filter_size=(5, 5),
          nonlinearity=lasagne.nonlinearities.rectify,
          W=lasagne.init.GlorotUniform())
  # Expert note: Lasagne provides alternative convolutional layers that
  # override Theano's choice of which implementation to use; for details
  # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

  # Max-pooling layer of factor 2 in both dimensions:
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

  # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
  network = lasagne.layers.Conv2DLayer(
          network, num_filters=32, filter_size=(5, 5),
          nonlinearity=lasagne.nonlinearities.rectify)
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

  # A fully-connected layer of 256 units with 50% dropout on its inputs:
  network = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=.5),
          num_units=256,
          nonlinearity=lasagne.nonlinearities.rectify)

  # And, finally, the 10-unit output layer with 50% dropout on its inputs:
  network = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=.5),
          num_units=102,
          nonlinearity=lasagne.nonlinearities.softmax)

  return network


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

def load_label(path):
    file = open(path, "r")
    label = []
    for line in file:
        token = line.split(",")
        label.append(token[1][:-1])
    return label


def main():
    # Load training and eval data

    encoder = LabelBinarizer()

  symbol_labels = load_label("data/DataSymbol_Iso/task2-trainSymb2014/trainingSymbols/iso_GT.txt")
  junk_labels = load_label("data/DataSymbol_Iso/task2-trainSymb2014/trainingJunk/junk_GT.txt")
  train_labels = symbol_labels + junk_labels[:74284]
  train_labels = np.asarray(train_labels)  
  y_train = encoder.fit_transform(train_labels)
  #num_classes = len(set(train_labels)) 
  if Path("train_data.npy").exists():
    X_train = np.load("train_data.npy")
  else: 
    X_train = load_images("data/DataSymbol_Iso/data/train")
    np.save("train_data", X_train)
  
  eval_labels = np.asarray(load_label("data/DataSymbol_Iso/task2-testSymbols2014/testSymbols_2016_iso_GT.txt"))
  y_test = encoder.fit_transform(eval_labels)
  if Path("eval_data.npy").exists():
    y_test = np.load("eval_data.npy")
  else:
    y_test = load_images("data/DataSymbol_Iso/data/test")
    np.save("eval_data", y_test)

  input_var = T.tensor4('inputs')
  target_var = T.ivector('targets')

  num_epochs=500

  network = build_cnn(input_var)

# Create a loss expression for training, i.e., a scalar objective we want
  # to minimize (for our multi-class problem, it is the cross-entropy loss):
  prediction = lasagne.layers.get_output(network)
  loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
  loss = loss.mean()
  # We could add some weight decay as well here, see lasagne.regularization.

  # Create update expressions for training, i.e., how to modify the
  # parameters at each training step. Here, we'll use Stochastic Gradient
  # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
  params = lasagne.layers.get_all_params(network, trainable=True)
  updates = lasagne.updates.nesterov_momentum(
          loss, params, learning_rate=0.01, momentum=0.9)

  # Create a loss expression for validation/testing. The crucial difference
  # here is that we do a deterministic forward pass through the network,
  # disabling dropout layers.
  test_prediction = lasagne.layers.get_output(network, deterministic=True)
  test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                          target_var)
  test_loss = test_loss.mean()
  # As a bonus, also create an expression for the classification accuracy:
  test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)

  # Compile a function performing a training step on a mini-batch (by giving
  # the updates dictionary) and returning the corresponding training loss:
  train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

  # Compile a second function computing the validation loss and accuracy:
  val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

  # Finally, launch the training loop.
  print("Starting training...")
  # We iterate over epochs:
  for epoch in range(num_epochs):
      # In each epoch, we do a full pass over the training data:
      train_err = 0
      train_batches = 0
      start_time = time.time()
      for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
          inputs, targets = batch
          train_err += train_fn(inputs, targets)
          train_batches += 1
  # After training, we compute and print the test error:
  test_err = 0
  test_acc = 0
  test_batches = 0
  for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
      inputs, targets = batch
      err, acc = val_fn(inputs, targets)
      test_err += err
      test_acc += acc
      test_batches += 1
  print("Final results:")
  print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
  print("  test accuracy:\t\t{:.2f} %".format(
      test_acc / test_batches * 100))
