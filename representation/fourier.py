import numpy
import trivial
import logging
import itertools


class FourierBasis(trivial.TrivialBasis):
    """Fourier Basis linear function approximation. Requires the ranges for each dimension, and is thus able to
    use only sine or cosine (and uses cosine). So, this has half the coefficients that a full Fourier approximation
    would use.

    From the paper:
    G.D. Konidaris, S. Osentoski and P.S. Thomas.
    Value Function Approximation in Reinforcement Learning using the Fourier Basis.
    In Proceedings of the Twenty-Fifth Conference on Artificial Intelligence, pages 380-385, August 2011.
    """

    def __init__(self, nvars, ranges, order=3):
        log = logging.getLogger('pyrl.representation.fourier')
        nterms = pow(order + 1, nvars)
        self.numTerms = nterms
        self.order = order
        self.ranges = numpy.array(ranges)
        iter = itertools.product(range(order + 1), repeat=nvars)
        self.multipliers = numpy.array([list(map(int, x)) for x in iter])
        log.debug("Numterms: %d   Order: %d \n Ranges: %s", self.numTerms, self.order, self.ranges)

    def computeFeatures(self, features):
        log = logging.getLogger('pyrl.representation.fourier.computeFeatures')
        if len(features) == 0:
            return numpy.ones((1,))
        basisFeatures = numpy.array([self.scale(features[i], i) for i in range(len(features))])
        log.info("Features: %s", features)
        log.info("Basis Features: %s", basisFeatures)
        # return_val = numpy.cos(numpy.pi * numpy.dot(self.multipliers, basisFeatures))
        return numpy.cos(numpy.pi * numpy.dot(self.multipliers, basisFeatures))  # return_val
