# Author: Will Dabney
# Author: Pierre-Luc Bacon <pierrelucbacon@gmail.com>

import csv, os
import logging

from misc.timer import Timer
from local_rlglue import RLGlueLocal as RLGlueLocal
from local_rlglue.registry import register_experiment
import rlglue.RLGlue as rl_glue


@register_experiment
class Episodic(object):
    name = "Episodic"

    def __init__(self, config, **kwargs):
        log = logging.getLogger('pyrl.experiment.episodic')

        self.maxsteps = kwargs.setdefault('maxsteps', 1000)  # do not change maxsteps here, use params to change it
        self.num_episodes = kwargs.setdefault('num_episodes', 200)  # same comment as above!
        self.num_runs = kwargs.setdefault('num_runs', 1)
        self.timed = kwargs.setdefault('timed', True)
        self.configuration = config

        log.info("Initializing experiment: %s", self.name)

        log.debug("Experiment params:")
        log.debug("Max steps: %d", self.maxsteps)
        log.debug("Number of episodes: %d", self.num_episodes)
        log.debug("Number of runs: %d", self.num_runs)
        log.debug("Timed run?: %s", self.timed)

        if kwargs.has_key('agent') and kwargs.has_key('environment'):
            log.info("Experiment in local mode")
            self.agent = kwargs['agent']
            self.environment = kwargs['environment']

            log.debug("Agent: %s", self.agent)
            log.debug("Environment: %s", self.environment)

            self.rlglue = RLGlueLocal.LocalGlue(self.environment, self.agent)
        else:
            log.info("Experiment in network mode")
            self.rlglue = rl_glue

    def run_episode(self):

        log = logging.getLogger('pyrl.experiment.episodic.runepisode')

        terminal = 0
        runtime = 0
        # Query the agent whether or not it has diverged
        if self.hasAgentDiverged():
            log.error("Agent diverged")
            return 0, -1, 0.0, 0.0  # -1 number of steps, signals that divergence.
        if self.timed:
            log.debug("Running episode in timed mode")
            timer = Timer()
            with timer:
                terminal = self.rlglue.RL_episode(self.maxsteps)
            runtime = timer.duration_in_seconds()
        else:
            terminal = self.rlglue.RL_episode(self.maxsteps)
        totalSteps = self.rlglue.RL_num_steps()
        totalReward = self.rlglue.RL_return()

        return terminal, totalSteps, totalReward, runtime

    def run_trial(self, filename=None):
        log = logging.getLogger('pyrl.experiment.episodic.runtrial')
        self.rlglue.RL_init()
        for i in range(self.num_episodes):
            term, steps, reward, runtime = self.run_episode()
            if filename is None:
                # print i, ',', steps, ',', runtime, ',', reward, ',', term
                # print str(i)+','+str(reward)
                print i, reward
            else:
                with open(filename, "a") as f:
                    csvwrite = csv.writer(f)
                    # csvwrite.writerow([i, steps, runtime, reward, term])
                    csvwrite.writerow([i, reward])
                    print i, reward
            log.info("Episode: %d, reward: %f, steps: %d, termination: %d", i, reward, steps, term)
            log.debug("Episode: %d  Reward: %f  Num steps: %d  Terminate: %d  Run time: %d", i, reward, steps, term,
                      runtime)
        self.rlglue.RL_cleanup()

    def run_experiment(self, filename=None):
        log = logging.getLogger('pyrl.experiment.episodic.runexperiment')

        if filename is None:
            log.info("No file name given. Writing results to console")
            print 'trial, reward'
            # print 'trial, number of steps, runtime, accumulated reward, termination'
        for run in range(self.num_runs):
            log.debug("Run: %d", run)
            print "Run:", run, "rewards"
            with open(filename, "a") as f:
                csvwrite = csv.writer(f)
                run_id = "Run: " + str(run)
                csvwrite.writerow([run_id, "rewards"])
            self.run_trial(filename=filename)

    def hasAgentDiverged(self):
        """Sends an rl-glue message to the agent asking if it has diverged or not.
        The message is exactly: agent_diverged?
        The expected response is: True (if it has), False (if it has not)
        The responses are not case sensitive, and anything other than true or false
        will be treated as a false (to support solvers which do not have this implemented).
        """
        return self.rlglue.RL_agent_message("agent_diverged?").lower() == "true"
