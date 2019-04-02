from collections import defaultdict
import numpy as np
import scipy

def choice_hack(data, p=None, size = 1):
    weights = p
    # all choices are at equal probability if no weights given
    if weights == None:
        weights = [1.0 / float(len(data)) for x in range(len(data))]
    if weights == None:
        weights = [1.0 / float(len(data)) for x in range(len(data))]

    if not np.sum(weights) == 1.0:
        if np.absolute(weights[0]) > 1.0e7 and sum(weights) == 0:
            weights = [1.0 / float(len(data)) for x in range(len(data))]
        else:
            raise Exception("Weights entered do not add up to 1! This must not happen!")

    # Compute edges of each bin
    edges = []
    etemp = 0.0
    for x, y in zip(data, weights):
        etemp = etemp + y
        edges.append(etemp)

    # if the np.sum of all weights does not add up to 1, raise an Exception
    if size == 1:
        randno = np.random.rand()

        # Else make sure that size is an integer
        # and make a list of random numbers
        try:
            randno = [np.random.rand() for x in np.arange(size)]
        except TypeError:
            raise TypeError("size should be an integer!")

        choice_index = np.array(edges).searchsorted(randno)
        choice_data = np.array(data)[choice_index]

        return choice_data



