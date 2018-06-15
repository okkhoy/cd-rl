# Author: Will Dabney
# Author: Pierre-Luc Bacon <pierrelucbacon@gmail.com>
# Author: Akshay Narayan

# added code to incorporate logging

# Runs an experiment by starting up rl_glue,
# and letting the user choose from a set of
# agents, environments, and experiments.

import json
import logging
import time
import os

from multiprocessing import Process
from subprocess import Popen

from agents import *
from environments import *
from experiments import *
from local_rlglue .registry import rlglue_registry
from misc.json import convert
from local_rlglue .myConfig import myConfig

from rlglue.agent import AgentLoader as AgentLoader
from rlglue.environment import EnvironmentLoader as EnvironmentLoader


def fromjson(filename):
    log = logging.getLogger('pyrl.run.fromjson')

    log.info("Loading configuration from experiment config file")

    with open(filename, 'r') as f:
        config = json.load(f, object_hook=convert)

    # Process the environment
    environment = rlglue_registry.environments[config['environment']['name']]
    environment_params = config['environment']['params']
    log.info("Environment: %s", environment)
    log.debug("Environment params: %s", environment_params)

    # Process the agent
    agent = rlglue_registry.agents[config['agent']['name']]
    agent_params = config['agent']['params']
    log.info("Agent: %s", agent)
    log.debug("Agent params: %s", agent_params)

    # Process the experiment
    experiment = rlglue_registry.experiments[config['experiment']['name']]
    experiment_params = config['experiment']['params']
    log.info("Experiment: %s", experiment)
    log.debug("Experiment params: %s", experiment_params)

    return agent, agent_params, environment, environment_params, experiment, experiment_params


def tojson(agent, a_args, env, env_args, exp, exp_args, local=None):
    config = {'agent': {'name': agent.name, 'params': a_args},
              'environment': {'name': env.name, 'params': env_args},
              'experiment': {'name': exp.name, 'params': exp_args}}
    return json.dumps(config)


def fromuser():
    log = logging.getLogger('pyrl.run.fromuser')

    log.info("Getting experiment configuration from user")

    environment = interactive_choose(rlglue_registry.environments,
                                     "Choose an environment.")
    agent = interactive_choose(rlglue_registry.agents, "Choose an agent.")
    experiment = interactive_choose(rlglue_registry.experiments,
                                    "Choose an experiment.")

    log.info("Environment: %s", environment)
    log.debug("Environment params: %s", '{}')

    log.info("Agent: %s", agent)
    log.debug("Agent params: %s", '{}')

    log.info("Experiment: %s", experiment)
    log.debug("Experiment params: %s", '{}')

    return agent, {}, environment, {}, experiment, {}


def interactive_choose(choices, prompt):
    print(prompt)
    sortkeys = sorted(choices.keys())

    for ix, a_key in enumerate(sortkeys):
        print("  ({:d}): {}".format(ix + 1, a_key))

    choice = None
    while choice not in range(1, len(sortkeys) + 1):
        choice = raw_input("Enter number (1 - {:d}): ".format(
            len(sortkeys)))
        try:
            choice = int(choice)
        except:
            pass

    return choices[sortkeys[choice - 1]]


def run(agent, a_args, env, env_args, exp, exp_args, local=None, result_file=None):
    log = logging.getLogger('pyrl.run.run')

    if local is None:
        ans = raw_input("Run locally? [y/n]: ")
        if ans.lower() == 'y' or ans.lower() == 'yes':
            local = True
        else:
            local = False

    config = {'agent': {'name': agent.name, 'params': a_args},
              'environment': {'name': env.name, 'params': env_args},
              'experiment': {'name': exp.name, 'params': exp_args}}
    # print "run method:\n", config, "\n"

    log.debug("Agent config: %s", config['agent'])
    log.debug("Environment config: %s", config['environment'])
    log.debug("Experiment config: %s", config['experiment'])

    if local:
        log.info("Running in local mode")

        experiment = exp(config, agent=agent(**a_args),
                         environment=env(**env_args), **exp_args)
        experiment.run_experiment(filename=result_file)
    else:
        log.info("Running in network mode")

        experiment = exp(config, **exp_args)
        # TODO: Figure out if rl_glue is running, don't start it in that case
        rlglue_p = Popen('rl_glue')
        agent_p = Process(target=AgentLoader.loadAgent,
                          args=(agent(**a_args),))
        agent_p.start()
        env_p = Process(target=EnvironmentLoader.loadEnvironment,
                        args=(env(**env_args),))
        env_p.start()
        experiment.run_experiment(filename=result_file, **a_args)
        env_p.terminate()
        agent_p.terminate()
        rlglue_p.terminate()


def addRunExpArgs(parser):
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument("--load", type=str, help="Load an experimental configuration from a JSON file.")
    json_group.add_argument("--genjson", action='store_true',
                            help="Generate an experimental configuration JSON file from " + \
                                 "interactive selections. Only generates, does not run.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action='store_true', default="True", help="Run experiment locally")
    group.add_argument("--network", action='store_true', help="Run experiment through network sockets")
    parser.add_argument("--output", type=str, help="Save the results to a file.")
    return parser


def prepare_logger():
    myConfig.logconfig = json.load(open("logconfig.json", 'r'))
    myConfig.start_time = get_time_now()
    if not os.path.exists(myConfig.logconfig['logfolder']):
        os.makedirs(myConfig.logconfig['logfolder'])
    logfilename = myConfig.logconfig['logfolder'] + "/" + myConfig.start_time + ".log"

    log = logging.getLogger('pyrl')
    logFile = logging.FileHandler(logfilename)
    log.setLevel(myConfig.logconfig['loglevel'])
    logFile.setFormatter(
        logging.Formatter('[%(asctime)s]: [%(filename)s:%(lineno)d:%(funcName)s]: %(levelname)s :: %(message)s',
                          datefmt='%m-%d-%Y %H:%M:%S'))
    log.addHandler(logFile)

    # print myConfig.logconfig['enabled']
    if myConfig.logconfig['enabled'] == False:
        log.info("Logging disabled. No more logs will be generated")
        logging.disable(logging.CRITICAL)

    log.info("New run: %s", myConfig.start_time)
    log.debug("Log configuration:")
    log.debug("Log file: %s", logfilename)
    log.debug("Log level: %s", myConfig.logconfig['loglevel'])
    log.debug("Log format: %s",
              '[\%(asctime)s]: [\%(filename)s:\%(lineno)d:\%(funcName)s]: \\\\n\\\\t \%(levelname)s :: \%(message)s')
    log.info("Logger ready")


def get_time_now():
    return time.strftime('%Y%m%d-%H%M', time.localtime())


if __name__ == '__main__':
    prepare_logger()

    log = logging.getLogger('pyrl')

    log.info("Run started")

    import argparse

    parser = argparse.ArgumentParser(
        description='Run a reinforcement learning experiment. Defaults to interactive experiment.')
    addRunExpArgs(parser)
    args = parser.parse_args()
    # print "Main:\n",  args, "\n"
    if args.load is None:
        log.info("No config file. Reading input from user")
        config = fromuser()
        if args.genjson:
            print (tojson(*config))
            # log.info("Writing configuration to json file. Config: %s", '; '.join("%s=%r" % (key,val) for (key,val) in config.iteritems()))
            log.debug("Writing configuration to json file. Config: %s", tojson(*config))
        else:
            # log.info("Starting experiment with user specified input. Config: %s", '; '.join("%s=%r" % (key,val) for (key,val) in config.iteritems()))
            log.debug("Running with user input. Config: %s", tojson(*config))
            run(*config, local=args.local, result_file=args.output)
    else:
        log.info("Config file specified")
        config = fromjson(args.load)
        log.info("Parsed config: %s", tojson(*config))
        # print "Parsed Config: \n", config, "\n"
        config_dict = json.loads(tojson(*config))
        op_agent_name = config_dict['agent']['name'].replace(" ", "")
        op_env_name = config_dict['environment']['name'].replace(" ", "")

        dumpPath = myConfig.logconfig['outputfolder'] + "/" + myConfig.start_time
        if not os.path.exists(dumpPath):
            os.makedirs(dumpPath)

        dumpFilePrefix = dumpPath + "/"

        output_filename = dumpFilePrefix + op_agent_name + "-" + \
                          op_env_name + "-" + myConfig.start_time + ".txt"

        run(*config, local=args.local, result_file=output_filename)  # args.output)
