from __future__ import division

from numpy import array, ndarray, ones, where
from numpy.random import random, randint


class WalkerRandom(object):
    '''
    Walker alias sampling method for random objects with different probabilities.
    In this context can use it for selecting actions from a preference list or selecting states.
    '''

    def __init__(self, weights, keys=None):
        '''
        Build Walker table with the given weights.
        Weights can be in any order need not sum to 1.
        '''

        if isinstance(weights, dict):
            keys = weights.keys()
            weights = weights.values()

        n = self.N = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = array(keys)

        if isinstance(weights, (list, tuple)):
            weights = array(weights, dtype=float)
        elif isinstance(weights, ndarray):
            if weights.dtype != float:
                weights.astype(float)
        else:
            weights = array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("Weights must be a vector")

        # get the probability of weights (numpy mehtod rocks!!!)
        weights = weights * n / weights.sum()
        inx = -ones(n, dtype=int)

        short = where(weights < 1)[0].tolist()
        long = where(weights > 1)[0].tolist()

        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= (1 - weights[j])

            if weights[k] < 1:
                short.append(k)
                long.pop()

        self.probability = weights
        self.inx = inx

    def randSample(self, count=None):
        '''
        Returns a given number of random integers / keys with probabilities proportional to
        the weights supplied in the constructor.

        When count == None, returns a single integer / key.
        Else, returns a NumPy array with a lenght given in count
        '''

        if count is None:
            u = random()
            j = randint(self.N)

            k = j if u <= self.probability[j] else self.inx[j]

            return self.keys[k] if self.keys is not None else k

        u = random(count)
        j = randint(self.N, size=count)
        k = where(u <= self.probability[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k
