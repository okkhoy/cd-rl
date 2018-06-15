# Created on April 3, 2017
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


import collections
import logging
import json
import random
import time
import copy
import os

from pprint import pformat

import MalmoPython

from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from local_rlglue import TaskSpecRLGlue
from local_rlglue.registry import register_environment

malmo_env = MalmoPython.AgentHost()
list_compare = lambda x, y: collections.Counter(x) == collections.Counter(y)


@register_environment
class SimpleMalmoEnvironment:
    name = 'SimpleMalmoEnvironment'


def __init__(self):
    log = logging.getLogger('SimpleMalmoEnvironment.init')

    # actions available for the agent
    self.actions = ["move 1", "turn 1", "turn -1", "use 1"]

    # mission elements: landmarks, size, landmark types, obstacles etc.,
    self.landmark_types = ["redstone_block", "emerald_block", "lapis_block", "gold_block", "cobblestone",
                           "quartz_block"]
    self.size = [6, 6]
    self.landmarks = [[1, 2], [2, 5], [5, 6], [5, 2]]
    self.obstacles = [[2, 2, 2], [3, 2, 2], [3, 3, 2], [4, 3, 2], [1, 4, 2], [2, 5, 2], [2, 6, 2]]

    # observation stuff that needs to be passed to the learning algorithm
    self.item_location = 0
    self.current_agent_location = []
    self.destination = 0
    self.last_observation = None
    self.last_action = ""
    self.direction = 0
    self.is_get_completed = False

    # malmo objects
    self.mission_xml = ""
    self.mission_record = None
    self.mission = None

    log.debug("Verify experiment config:\n%s", pformat(self.__dict__))


def generate_malmo_environment_xml(self):
    log = logging.getLogger('SimpleMalmoEnvironment.generateMalmoEnvironmentXML')
    xml_string = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            <About>
                <Summary>Simple Malmo Environment</Summary>
            </About>
            <ServerSection>
                <ServerInitialConditions>
                    <Time><StartTime>1</StartTime></Time>
                    <Weather>clear</Weather>
                </ServerInitialConditions>
                <ServerHandlers>
                    <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" forceReset="true" />
                <DrawingDecorator>
                    <!-- coordinates for cuboid are inclusive -->
                    <DrawCuboid x1="0" y1="46" z1="0" x2="''' + str(self.size[0]) + '''" y2="50" z2="''' + str(
        self.size[1]) + '''" type="air" />            <!-- limits of our arena -->
                    <DrawCuboid x1="0" y1="45" z1="0" x2="''' + str(self.size[1]) + '''" y2="45" z2="''' + str(
        self.size[1]) + '''" type="sandstone" />      <!-- floor of the arena -->
                    ''' + self.draw_landmarks() + self.draw_obstacles() + '''
                </DrawingDecorator>
                <ServerQuitFromTimeUp timeLimitMs="50000"/>
                <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
            </ServerSection>
            <AgentSection mode="Survival">
                <Name>ButterFingers</Name>
                <AgentStart>
                    <Inventory>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <MissionQuitCommands/>
                    <InventoryCommands/>
                    <ObservationFromFullStats/>
                    <ObservationFromRay/>
                    <ObservationFromFullInventory/>
                    <ObservationFromDiscreteCell/>
                    <VideoProducer want_depth="false" viewpoint="1">
                        <Width>480</Width>
                        <Height>320</Height>
                    </VideoProducer>
                    <RewardForSendingCommand reward="-1" />
                </AgentHandlers>
                </AgentSection>
        </Mission>'''

    log.debug("Final mission XML String: \n%s", xml_string)

    return xml_string


def draw_landmarks(self):
    log = logging.getLogger('SimpleMalmoEnvironment.drawLandmarks')

    landmarks = self.landmarks

    log.debug("Input landmarks: %s", landmarks)

    landmark_xml = '''<!-- draw landmarks -->'''

    for i, l in enumerate(landmarks):
        x, z = l
        landmark_xml += '''<DrawBlock x="''' + str(x) + '''" y="45" z="''' + str(z) + '''" type="''' + \
                        self.landmark_types[i % len(self.landmark_types)] + '''"/>'''

    return landmark_xml


def draw_obstacles(self):
    log = logging.getLogger('SimpleMalmoEnvironment.drawObstacles')

    log.debug("Input walls: %s", self.obstacles)

    obstacle_xml = '''<!-- draw the obstacles -->'''
    for w in self.obstacles:
        log.debug("Adding obstacle: %s", w)
        x, y, direction = list(w)
        obstacle_string = self.draw_obstacle(x, y, direction)
        obstacle_xml = obstacle_xml + obstacle_string

    log.debug("Obstacle string obtained: %s", obstacle_xml)

    return obstacle_xml


def draw_obstacle(self, x, y, direction):
    _x, _y = x, y

    log = logging.getLogger('SimpleMalmoEnvironment.drawObstacle')

    log.debug("Input x, y, d: %d, %d, %d", x, y, direction)

    if direction == 3 and x == 0:
        log.info("No wall added")
        log.debug("Trying to add wall on the left boundary.")
        return ""
    if direction == 2 and x == self.size[0] - 1:
        log.info("No wall added")
        log.debug("Trying to add wall on the right boundary.")
        return ""
    if direction == 1 and y == 0:
        log.info("No wall added")
        log.debug("Trying to add wall on the bottom boundary.")
        return ""
    if direction == 0 and y == self.size[1] - 1:
        log.info("No wall added")
        log.debug("Trying to add wall on the top boundary.")
        return ""

    if direction == 0:  # north bit value: 0001
        _y = y + 1

    elif direction == 1:  # south bit value: 0010
        _y = y - 1

    elif direction == 2:  # east bit value: 0100
        _x = x + 1

    elif direction == 3:  # west bit value: 1000
        _x = x - 1

    # clip the values between the max and min environment size.
    _x = 0 if _x < 0 else self.size[0] if _x > self.size[0] else _x
    _y = 0 if _y < 0 else self.size[0] if _y > self.size[0] else _y

    obstacle_string = '''<DrawBlock x="''' + str(_x) + '''" y="45" z="''' + str(_y) + '''"  type="bedrock" />
                    <DrawBlock x="''' + str(_x) + '''" y="46" z="''' + str(_y) + '''"  type="bedrock" />
                    <DrawBlock x="''' + str(_x) + '''" y="47" z="''' + str(_y) + '''"  type="bedrock" />
                    <DrawBlock x="''' + str(_x) + '''" y="48" z="''' + str(_y) + '''"  type="beacon" />
                    '''

    log.debug("Obstacle string: %s", obstacle_string)
    return obstacle_string


def makeObservation(self, action_status=False):
    log = logging.getLogger('SimpleMalmoEnvironment.makeObservation')

    target_item = self.landmark_types[self.destination]

    # get the state from malmo
    world_state = malmo_env.getWorldState()

    terminal = 0
    # wait for first valid observation to be received
    while True:
        time.sleep(0.1)
        world_state = malmo_env.getWorldState()

        for error in world_state.errors:
            self.logger.error("Error: %s" % error.text)
        log.debug("Received: %s, Num. observations: %d", world_state, len(world_state.observations))
        if world_state.is_mission_running and len(world_state.observations) > 0 \
                and not world_state.observations[-1].text == "{}":
            observation = json.loads(world_state.observations[-1].text)
            if not u'XPos' in observation or not u'ZPos' in observation or not u'Yaw' in observation \
                    or not u'LineOfSight' in observation:
                # hack to force an observation; sometimes the world_state.observations may not have LineOfSight
                # this line below forces the environment to return an observation.
                malmo_env.sendCommand("jump 0")
                log.error("Incomplete observation received: %s", pformat(observation))
            else:
                self.last_observation = copy.deepcopy(observation)
                break
        if not world_state.is_mission_running:
            terminal = 1
            break

    if not world_state.is_mission_running:
        terminal = 1
        return_observation = {}
        log.warn("Malmo mission ended")

    current_r = 0
    for reward in world_state.rewards:
        current_r += reward.getValue()

    if world_state.is_mission_running:
        log.debug("Received world state: %s", world_state)
        log.debug("Received observation %s", pformat(observation))

        x, y = int(observation[u'XPos']), int(observation[u'ZPos'])
        self.direction = int(observation[u'Yaw']) / 90

        if not self.is_get_completed:
            self.is_get_completed = self.check_inventory(observation, target_item)

            if self.is_get_completed:
                self.item_location = len(self.landmarks)
                current_r += 11
                log.debug("Get successfully completed")
        elif self.is_get_completed:
            if "use" in self.last_action:
                if action_status is True:
                    current_r += 21
                    log.debug("Put successfully completed")
                    terminal = 1
                else:
                    current_r -= 10
                    log.debug("Put failed")

        distance = float(observation[u'LineOfSight'][u'distance'])
        return_observation = {"intobs": [x, y, self.direction, self.item_location, self.destination],
                              "floatobs": [distance]}

    return return_observation, current_r, terminal


def reset(self):
    log = logging.getLogger('SimpleMalmoEnvironment.reset')

    obstacle_locations = [[l[0], l[1]] for l in self.obstacles]
    landmark_locations = [[l[0], l[1]] for l in self.landmarks]

    del self.mission  # just to be sure, i create a new mission every episode
    # mission related objects
    self.mission_xml = self.generate_malmo_environment_xml()
    log.debug("Obtained mission XML: \n %s", self.mission_xml)
    self.mission_record = MalmoPython.MissionRecordSpec()
    self.mission = MalmoPython.MissionSpec(self.mission_xml, True)
    log.info("Loaded mission XML")

    # select a random start location such that is is not one of the wall cells and not one of landmarks
    # x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
    # while [x, y] in obstacle_locations or [x, y] in landmark_locations:
    #     x, y = random.randint(0, 6), random.randint(0, 6)

    self.mission.setViewpoint(1)

    # set mission variables - landmarks, source and destination
    landmarks = copy.deepcopy(self.landmarks)
    source_loc = random.choice(landmarks)  # first select the source to pick up from
    remaining_landmarks = [lm for lm in landmarks if lm != source_loc]  # tentative destinations are other landmarks
    destination = random.choice(remaining_landmarks)  # now randomly choose the destination from above list
    agent_start_loc = random.choice(remaining_landmarks)  # start locations for agent; start loc != pick up source
    x, y = agent_start_loc[0], agent_start_loc[1]
    self.current_agent_location = [x, y]
    # malmo needs locations to be 0.5 to be in the middle of the square, else, it is at the edge
    self.mission.startAt(x + 0.5, 46, y + 0.5)

    self.item_location = landmarks.index(source_loc)
    self.destination = landmarks.index(destination)

    self.mission.drawItem(source_loc[0], 47, source_loc[1], self.landmark_types[self.destination])

    retries = 3
    log.debug("Final Mission XML sent to Malmo: \n %s", self.mission.getAsXML(True))
    for retry in range(retries):
        try:
            malmo_env.startMission(self.mission, self.mission_record)
            time.sleep(10)

            world_state = malmo_env.getWorldState()
            if world_state.has_mission_begun:
                break
        except RuntimeError as e:
            if retry == retries - 1:
                log.error("Error starting mission. Max retries elapsed. Closing! %s", e.message)
                exit(1)
            else:
                time.sleep(10)

    world_state = malmo_env.getWorldState()

    while not world_state.has_mission_begun:
        log.debug("Waiting for mission to begin")
        time.sleep(0.1)
        world_state = malmo_env.getWorldState()
        for error in world_state.errors:
            log.error("Error: %s", error.text)


def env_start(self):
    log = logging.getLogger('SimpleMalmoEnvironment.envStart')

    # need to quit before you start a new mission, else it fails below with runtime error
    log.debug("Sending quit command to restart the mission")
    malmo_env.sendCommand("quit")

    self.reset()
    log.info("Environment started")

    return_observation, reward, terminal = self.makeObservation()
    log.debug("First observation: %s, %f, %d", return_observation, reward, terminal)

    return return_observation


def env_step(self, thisAction):
    log = logging.getLogger('SimpleMalmoEnvironment.envStep')

    log.debug("Received action: %s", str(thisAction))

    malmo_action = self.actions[thisAction]

    self.last_action = malmo_action

    action_status = self.take_action(malmo_action)

    obs, reward, terminal = self.makeObservation(action_status)

    log.debug("Observation: %s ; reward = %f ; terminal = %d", pformat(obs), reward, terminal)

    return obs, reward, terminal


def take_action(self, malmo_action):
    log = logging.getLogger('SimpleMalmoEnvironment.takeAction')

    log.debug("Action to take: %s", malmo_action)

    # get the target landmark from the list
    landmark_types = self.landmark_types
    target_block = landmark_types[self.destination]

    if "use" in malmo_action:
        log.debug("Action to put things down")

        retry_for_obs = 5
        trial = 0
        while trial < retry_for_obs:
            trial += 1
            time.sleep(0.1)
            world_state = malmo_env.getWorldState()

            for error in world_state.errors:
                log.error("Error: %s", error.text)

            log.debug("Received: %s, Num. observations: %d", world_state, len(world_state.observations))

            if world_state.is_mission_running and len(world_state.observations) > 0 \
                    and not world_state.observations[-1].text == "{}":
                observation = json.loads(world_state.observations[-1].text)

                if not u'XPos' in observation or not u'ZPos' in observation or not u'Yaw' in observation \
                        or not u'LineOfSight' in observation:
                    malmo_env.sendCommand("jump 0")
                    log.error("Incomplete observation received: %s", pformat(observation))
                    log.warn("Sending a noop jump 0 command to force observation")
                else:
                    log.debug("Put: Got it!!: : : Observation in put: %s", pformat(observation))
                    break

        x, y = int(observation[u'XPos']), int(observation[u'ZPos'])
        dest = self.landmarks[self.destination]

        log.debug("Currently at: %d, %d", x, y)
        log.debug("Destination: %s", dest)

        if not list_compare([x, y], dest):
            log.warn("Not correct destination, put down action failed. Get negative reward")
            return False
        else:
            log.info("Can put down now [%d,%d]", x, y)

    try:
        malmo_env.sendCommand(malmo_action)
        log.info("Action %s succeeded", malmo_action)
    except RuntimeError as e:
        log.error("Failed to send command %s", e)
        return False

    return True


def check_inventory(self, observation, required):
    # need to find a way to see if the get task has been completed. one of the ways to do it is to check the
    # inventory, if the block has been acquired.
    for i in xrange(0, 39):
        key = 'InventorySlot_' + str(i) + '_item'
        if key in observation:
            item = observation[key]
            if item == required:
                return True

    return False


def main():
    prepare_logger()
    log = logging.getLogger('SimpleMalmoEnvironment')
    log.info('Start!')
    malmo = SimpleMalmoEnvironment()

    n_steps_per_ep = 500
    n_episodes = 100
    n_runs = 10

    for r in xrange(n_runs):
        for e in xrange(n_episodes):
            # start the episode
            malmo.env_start()

            terminal = False
            i = 0
            actions = range(len(malmo.actions))
            # loop until episode ends
            while i < n_steps_per_ep and not terminal:
                log.debug("Run: %d; Episode: %d; Step: %d", r, e, i)
                i += 1
                if not malmo.is_get_completed:
                    action = random.choice(actions)
                else:
                    action = random.choice(actions[0:3])
                obs, reward, terminal = malmo.env_step(action)


def get_time_now():
    return time.strftime('%Y%m%d-%H%M', time.localtime())


def prepare_logger():
    start_time = get_time_now()
    if not os.path.exists('log'):
        os.makedirs('log')

    logfilename = 'log/' + start_time + '.log'
    log = logging.getLogger('SimpleMalmoEnvironment')
    logFile = logging.FileHandler(logfilename)
    log.setLevel('DEBUG')
    logFile.setFormatter(
        logging.Formatter('[%(asctime)s]: [%(filename)s:%(lineno)d:%(funcName)s]: %(levelname)s :: %(message)s',
                          datefmt='%m-%d-%Y %H:%M:%S'))
    log.addHandler(logFile)

    log.info("Logger ready")


if __name__ == '__main__':
    main()
