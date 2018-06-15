# Created on May 7, 2017
#
# @author: Akshay Narayan
#
# This code is shared under The MIT License
# -----------------------------------------
#
# The MIT License (MIT)
#
# Copyright (c) <year> <copyright holders>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import numpy as np
import random


class EpsilonGreedy:

    def __init__(self, epsilon, counts, values):
        """
        :param epsilon: tells the probability with which we explore 
        :param counts: vector of length N that tells how many times we played each of N arms
        :param values: vector of length N that tells the average amount of rewards obtained playing each of N arms
        """
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        """
        :param n_arms: tells the number of arms (or actions) in the problem 
        """
        self.counts = [0 for col in xrange(n_arms)]
        self.values = [0.0 for col in xrange(n_arms)]

    def set_epsilon(self, epsilon):
        """
        Setter method
        :param epsilon: the value of epsilon to be set for this instance of the algorithm
        """
        self.epsilon = epsilon

    def select_arm(self):
        """
        :return: the index of the arm to be pulled (action to be performed)
        """
        if random.random() > self.epsilon:
            return index_max(self.values)
        else:
            random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        """
        Increments count of the chosen arm.
        Determines the current estimated value of the chosen arm.
            Updates the estimated value of the chosen arm to be a weighted average of the
            previously estimated value and the reward just received.
            Update rule actually counts the running average
        :param chosen_arm: arm that was selected to be pulled (action to be performed)
        :param reward: numerical value obtained for performing the action
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


def index_max(x):
    """
    :param x: vector whose max index has to be returned 
    :return: index of the max element in x
    """
    m = max(x)
    return x.index(m)


def test():
    algo = EpsilonGreedy(1.0, [])
    algo.initialize(2)
