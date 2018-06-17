from random import Random
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

import logging
import numpy
import time
import copy
import sys

from pprint import pformat

from local_rlglue.registry import register_agent
from skeleton_agent import *
import representation.fourier as fourier
import representation.trivial as trivial
import stepsizes


@register_agent
class sarsa_lambda(skeleton_agent):
    name = "Sarsa"
    policy = {}

    def init_parameters(self):
        # Initialize algorithm parameters
        log = logging.getLogger('pyrl.agents.sarsa_lambda')
        self.epsilon = self.params.setdefault('epsilon', 0.01)
        self.alpha = self.params.setdefault('alpha', 0.01)
        self.lmbda = self.params.setdefault('lmbda', 0.7)  # no elgibility traces
        self.gamma = self.params.setdefault('gamma', 0.9)
        self.fa_name = self.params.setdefault('basis', 'trivial')
        self.softmax = self.params.setdefault('softmax', False)
        self.basis = None
        log.debug("Sarsa Lambda: %s", pformat(self.__dict__))

    @classmethod
    def agent_parameters(cls):
        param_set = super(sarsa_lambda, cls).agent_parameters()
        add_parameter(param_set, "alpha", default=0.01, help="Step-size parameter")
        add_parameter(param_set, "epsilon", default=0.1,
                      help="Exploration rate for epsilon-greedy, or rescaling factor for soft-max.")
        add_parameter(param_set, "gamma", default=1.0, help="Discount factor")
        add_parameter(param_set, "lmbda", default=0.7, help="Eligibility decay rate")

        # Parameters *NOT* used in parameter optimization
        add_parameter(param_set, "softmax", optimize=False, type=bool, default=False, help="Use soft-max policies")
        add_parameter(param_set, "basis", optimize=False, type=str,
                      help="Basis to use with linear function approximation",
                      choices=['trivial', 'fourier', 'rbf', 'tile'], default='trivial')
        add_parameter(param_set, "fourier_order", optimize=False, default=3, type=int, min=1, max=15)
        add_parameter(param_set, "rbf_number", optimize=False, default=0, type=int, min=0, max=500)
        add_parameter(param_set, "rbf_beta", optimize=False, default=0.9)
        add_parameter(param_set, "tile_number", optimize=False, default=100, type=int, min=0, max=500)
        add_parameter(param_set, "tile_weights", optimize=False, default=2 ** 11, type=int, min=1, max=2 ** 15)
        return param_set

    def agent_supported(self, parsedSpec):
        if parsedSpec.valid:
            # Check observation form, and then set up number of features/states
            assert len(parsedSpec.getDoubleObservations()) + len(
                parsedSpec.getIntObservations()) > 0, "Expecting at least one continuous or discrete observation"

            # Check action form, and then set number of actions
            assert len(parsedSpec.getIntActions()) == 1, "Expecting 1-dimensional discrete actions"
            assert len(parsedSpec.getDoubleActions()) == 0, "Expecting no continuous actions"
            assert not parsedSpec.isSpecial(
                parsedSpec.getIntActions()[0][0]), "Expecting min action to be a number not a special value"
            assert not parsedSpec.isSpecial(
                parsedSpec.getIntActions()[0][1]), "Expecting max action to be a number not a special value"
            return True
        else:
            return False

    def agent_init(self, taskSpec):
        """Initialize the RL agent.

        Args:
            taskSpec: The RLGlue task specification string.
        """

        # (Re)initialize parameters (incase they have been changed during a trial
        log = logging.getLogger('pyrl.agents.sarsa_lambda.agent_init')
        self.init_parameters()
        # Parse the task specification and set up the weights and such
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
        if not self.agent_supported(TaskSpec):
            print "Task Spec could not be parsed: " + taskSpec;
            sys.exit(1)

        self.numStates = len(TaskSpec.getDoubleObservations())
        log.info("Ranges: %s", TaskSpec.getDoubleObservations())
        self.discStates = numpy.array(TaskSpec.getIntObservations())
        self.numDiscStates = int(reduce(lambda a, b: a * (b[1] - b[0] + 1), self.discStates, 1.0))
        self.numActions = TaskSpec.getIntActions()[0][1] + 1

        # print "TSactions ", TaskSpec.getIntActions(), "TSObservation ", TaskSpec.getIntObservations()

        if self.numStates == 0:
            # Only discrete states
            self.numStates = 1
            if self.fa_name != "trivial":
                print "Selected basis requires at least one continuous feature. Using trivial basis."
                self.fa_name = "trivial"

        # Set up the function approximation
        if self.fa_name == 'fourier':
            self.basis = fourier.FourierBasis(self.numStates, TaskSpec.getDoubleObservations(),
                                              order=self.params.setdefault('fourier_order', 3))
        else:
            self.basis = trivial.TrivialBasis(self.numStates, TaskSpec.getDoubleObservations())

        log.debug("Num disc states: %d", self.numDiscStates)
        numStates = self.basis.getNumBasisFunctions()
        log.debug("Num states: %d", numStates)
        log.debug("Num actions: %d", self.numActions)
        self.weights = numpy.zeros((self.numDiscStates, numStates, self.numActions))
        self.traces = numpy.zeros(self.weights.shape)
        self.init_stepsize(self.weights.shape, self.params)
        # print "Weights:", self.weights
        self.lastAction = Action()
        self.lastObservation = Observation()
        log.debug("Sarsa Lambda agent after initialization: %s", pformat(self.__dict__))

    def getAction(self, state, discState):
        """Get the action under the current policy for the given state.

        Args:
            state: The array of continuous state features
            discState: The integer representing the current discrete state value

        Returns:
            The current policy action, or a random action with some probability.
        """
        log = logging.getLogger('pyrl.agents.sarsa_lambda.getAction')
        if self.softmax:
            log.debug("Softmax enabled")
            softmax_action = self.sample_softmax(state, discState)
            log.debug("Action to return: %d", softmax_action)
            return softmax_action
        else:
            log.debug("Softmax disabled -- using e-greedy")
            egreedy_action = self.egreedy(state, discState)
            log.debug("Action to return: %d", egreedy_action)
            return egreedy_action

    def sample_softmax(self, state, discState):
        log = logging.getLogger('pyrl.agents.sarsa_lambda.sample_softmax')
        log.debug("weights: %s", self.weights[discState, :, :].T)
        log.debug("State: %s", state)
        Q = None
        Q = numpy.dot(self.weights[discState, :, :].T, self.basis.computeFeatures(state))
        Q -= Q.max()
        Q = numpy.exp(numpy.clip(Q / self.epsilon, -500, 500))
        Q /= Q.sum()

        Q = Q.cumsum()
        return_action = numpy.where(Q >= numpy.random.random())[0][0]
        log.debug("Return action: %d", return_action)
        return return_action

    def egreedy(self, state, discState):
        log = logging.getLogger('pyrl.agents.sarsa_lambda.egreedy')
        if self.randGenerator.random() < self.epsilon:
            rand_action = self.randGenerator.randint(0, self.numActions - 1)
            log.debug("Random action: %d", rand_action)
            return rand_action

        log.debug("Weights: %s", self.weights[discState, :, :].T)
        return_action = numpy.dot(self.weights[discState, :, :].T, self.basis.computeFeatures(state)).argmax()
        log.debug("Return action: %d", return_action)
        return return_action

    def getDiscState(self, state):
        """Return the integer value representing the current discrete state.

        Args:
            state: The array of integer state features

        Returns:
            The integer value representing the current discrete state
        """

        if self.numDiscStates > 1:
            x = numpy.zeros((self.numDiscStates,))
            # print "nDiscSt", self.numDiscStates, "x:", x, " ",self.discStates, " asdas ",
            mxs = self.discStates[:, 1] - self.discStates[:, 0] + 1
            # print "mxs:", mxs
            mxs = numpy.array(list(mxs[:0:-1].cumprod()[::-1]) + [1])
            # print "mxsMod:", mxs, " ",
            x = numpy.array(state) - self.discStates[:, 0]
            # print "X:", x, " DiscST :",
            # print (x*mxs).sum()
            return (x * mxs).sum()
        else:
            return 0

    def agent_start(self, observation):
        """Start an episode for the RL agent.

        Args:
            observation: The first observation of the episode. Should be an RLGlue Observation object.

        Returns:
            The first action the RL agent chooses to take, represented as an RLGlue Action object.
        """
        log = logging.getLogger('pyrl.agents.sarsa_lambda.agent_start')
        theState = numpy.array(list(observation.doubleArray))
        thisIntAction = self.getAction(theState, self.getDiscState(observation.intArray))

        returnAction = Action()
        returnAction.intArray = [thisIntAction]

        # Clear traces
        self.traces.fill(0.0)

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)
        log.debug("Action: %d", thisIntAction)
        log.debug("Start State: %s", theState)
        log.debug("Traces: %s", self.traces)
        return returnAction

    def update_traces(self, phi_t, phi_tp):
        log = logging.getLogger('pyrl.agents.sarsa_lambda.update_traces')
        log.debug("\nphi_t: %s \n phi_tp: %s", phi_t, phi_tp)
        self.traces *= self.gamma * self.lmbda
        self.traces += phi_t
        log.debug("Traces: %s", self.traces)

    def agent_step(self, reward, observation):
        """Take one step in an episode for the agent, as the result of taking the last action.

        Args:
            reward: The reward received for taking the last action from the previous state.
            observation: The next observation of the episode, which is the consequence of taking the previous action.

        Returns:
            The next action the RL agent chooses to take, represented as an RLGlue Action object.
        """
        log = logging.getLogger('pyrl.agents.sarsa_lambda.agent_step')
        newState = numpy.array(list(observation.doubleArray))
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]

        newDiscState = self.getDiscState(observation.intArray)
        lastDiscState = self.getDiscState(self.lastObservation.intArray)
        newIntAction = self.getAction(newState, newDiscState)

        self.policy.update({lastDiscState: lastAction})
        # Update eligibility traces
        phi_t = numpy.zeros(self.traces.shape)
        phi_tp = numpy.zeros(self.traces.shape)
        phi_t[lastDiscState, :, lastAction] = self.basis.computeFeatures(lastState)
        phi_tp[newDiscState, :, newIntAction] = self.basis.computeFeatures(newState)

        self.update_traces(phi_t, phi_tp)
        self.update(phi_t, phi_tp, reward)

        returnAction = Action()
        returnAction.intArray = [newIntAction]

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)
        # print "new state:", newDiscState,
        # print "last state:", lastDiscState
        log.debug("Last State: %s", lastState)
        log.debug("Last Action: %d", lastAction)
        log.debug("New Action: %d", newIntAction)
        log.debug("Current State: %s", newState)
        log.debug("Traces: %s", self.traces)
        log.debug("Weights: %s", self.weights)

        return returnAction

    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        return self.step_sizes * descent_direction

    def update(self, phi_t, phi_tp, reward):
        # Compute Delta (TD-error)
        delta = numpy.dot(self.weights.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward

        # Update the weights with both a scalar and vector stepsize used
        # Adaptive step-size if that is enabled
        self.weights += self.rescale_update(phi_t, phi_tp, delta, reward, delta * self.traces)

    def agent_end(self, reward):
        """Receive the final reward in an episode, also signaling the end of the episode.

        Args:
            reward: The reward received for taking the last action from the previous state.
        """
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]

        lastDiscState = self.getDiscState(self.lastObservation.intArray)

        # Update eligibility traces
        phi_t = numpy.zeros(self.traces.shape)
        phi_tp = numpy.zeros(self.traces.shape)
        phi_t[lastDiscState, :, lastAction] = self.basis.computeFeatures(lastState)

        self.update_traces(phi_t, phi_tp)
        self.update(phi_t, phi_tp, reward)
        self.policy.update({lastDiscState: lastAction})

    # for key in sorted(self.policy.keys()):
    #	print key, ":", self.policy[key], "\t",

    def agent_cleanup(self):
        """Perform any clean up operations before the end of an experiment."""
        pass

    def has_diverged(self):
        value = self.weights.sum()
        return numpy.isnan(value) or numpy.isinf(value)


# @register_agent
class residual_gradient(sarsa_lambda):
    """Residual Gradient(lambda) algorithm. This RL algorithm is essentially what Sarsa(labmda)
    would be if you were actually doing gradient descent on the squared Bellman error.

    From the paper (original):
    Residual Algorithms: Reinforcement Learning with Function Approximation.
    Leemon Baird. 1995.
    """

    name = "Residual Gradient"

    def update_traces(self, phi_t, phi_tp):
        self.traces *= self.gamma * self.lmbda
        self.traces += (phi_t - self.gamma * phi_tp)


# @register_agent
class fixed_policy(sarsa_lambda):
    """This agent takes a seed from which it generates the weights for the
    state-action value function. It then behaves just like Sarsa but with a
    learning rate of zero (0). Thus, it has a fixed state-action value function
    and thus a fixed policy (which has been randomly generated).
    """

    name = "Fixed Policy"

    def init_parameters(self):
        sarsa_lambda.init_parameters(self)
        self.policy_seed = self.params.setdefault('seed', int(time.time() * 10000))

    @classmethod
    def agent_parameters(cls):
        param_set = super(fixed_policy, cls).agent_parameters()
        add_parameter(param_set, "seed", type=int, default=int(time.time() * 10000), min=1, max=int(1.4e13))
        return param_set

    def agent_init(self, taskSpec):
        sarsa_lambda.agent_init(self, taskSpec)
        numpy.random.seed(self.policy_seed)
        self.weights = 2. * (numpy.random.random(self.weights.shape) - 0.5)
        numpy.random.seed(None)

    def update(self, phi_t, phi_tp, reward):
        pass


ABSarsa = stepsizes.genAdaptiveAgent(stepsizes.AlphaBounds, sarsa_lambda)

if __name__ == "__main__":
    runAgent(sarsa_lambda)
