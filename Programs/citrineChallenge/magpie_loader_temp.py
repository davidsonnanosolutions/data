"""
magpie_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import random
import ast

# Third-party libraries
import numpy as np
import pandas as pd

global training_data_start
global training_data_end
global results_data_position

training_data_start, training_data_end, results_data_position = [2,-1,-1]

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    with open('/home/wizard/data/citirineChallenge/training_data.csv', 'rb') as f:

        raw_df = pd.read_csv(f)
        
        valIndex = random.sample(xrange(len(raw_df)),int(round(0.1*len(raw_df))))
        validation_df = pd.DataFrame(raw_df.ix[i,2:-1] for i in valIndex).sort_index(ascending=True)
        validation_df.reset_index()

        input_df = raw_df.drop(valIndex)
        input_df.reset_index()

        input_data = []
        input_data.append(input_df.iloc[:,training_data_start:training_data_end])
        input_data.append(input_df.iloc[:,results_data_position])

        validation_data = []
        validation_data.append(validation_df.iloc[:,training_data_start:training_data_end])
        validation_data.append(validation_df.iloc[:,results_data_position])

    with open('/home/wizard/data/citirineChallenge/test_data.csv', 'rb') as f:

        test_df = pd.read_csv(f)

        test_data = []
        test_data.append(test_df.iloc[:,training_data_start:training_data_end])
        test_data.append(test_df.iloc[:,results_data_position])

    return(input_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    tr_d, va_d, te_d = load_data()

    training_inputs = [np.ndarray(tr_d[0].iloc[row]).astype('float32') for row in range(0,len(tr_d[0]))]
    training_results = [np.ndarray(vectorize(tr_d[1].iloc[row])) for row in range(0, len(tr_d[1]))]
    training_data = zip(training_inputs, training_results)

    validation_data = [np.ndarray(va_d[0].iloc[row]).astype('float32') for row in range(0,len(va_d[0]))]
    
    test_data = [np.ndarray(te_d[0].iloc[row]).astype('float32') for row in range(0,len(te_d[0]))]

    print 'Training Inputs: {} '.format(len(training_data))
    return (training_data, validation_data, test_data)

def vectorize(d):
    d = ast.literal_eval(d)
    e = np.zeros((11, 1))
    for element in xrange(0,len(d)):
        e[element] = d[element]
    return(e)