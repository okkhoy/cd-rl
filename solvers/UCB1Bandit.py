# Created on May 8, 2017
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


import math


class UCB1:
    def __init__(self, counts, values):
        """ 
        :param counts: vector of length N that tells how many times we played each of N arms
        :param values: vector of length N that tells the average amount of rewards obtained playing each of N arms
        """
        self.counts = counts
        self.vaues = values

    def initialize(self, n_arms):
        """
        :param n_arms: tells the number of arms (or actions) in the problem 
        """
        self.counts = [0 for col in xrange(n_arms)]
        self.values = [0.0 for col in xrange(n_arms)]

    def select_arm(self):
        """
        :return: the index of the arm to be pulled (action to be performed)
        """
        n_arms = len(self.counts)

        for arm in xrange(n_arms):
            if self.counts[arm] == 0:  # ensure that every arm is played at least once
                return arm

        # if number of trials is less than the number of arms, then each arm may never be
        # tried out before making the optimal decision.

        ucb_values = [0.0 for arm in xrange(n_arms)]
        total_counts = sum(self.counts)

        for arm in xrange(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            # a measure of how much less we know; when counts[arm] is low, the fraction becomes
            # large; hence the bonus value is high. index max below selects arm with highest value
            ucb_values[arm] = self.values[arm] + bonus

        return index_max(ucb_values)

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
