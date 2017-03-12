'''
Created on 17.02.2017

@author: Martin
'''

#--- enable this to run on GPU
#import os
#os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"
#---

import random
import numpy as np
import math
from SumTree import SumTree
import pygame
from Preprocessor import CNNPreprocessor
from RL_Algo import Brain
import RL_Algo
from SimpleGame import Learn_SinglePlayer

from keras import backend as K
K.set_image_dim_ordering('th')


""" Double "Deep Q -Network" with PER """
SIZE = 10
DEPTH = 1
STATE_CNT = (DEPTH, 1,SIZE)
ACTION_CNT = 2  # left, right, straight

MEMORY_CAPACITY = 200  # change to 200 000 (1 000 000 in original paper)

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1

# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 500000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

SAVE_XTH_GAME = 1000 # all x games, save the CNN
LEARNING_FRAMES = 1000000 # 50mio

#-------------------- BRAIN ---------------------------


class DQN_Brain(Brain):

    def __init__(self):

        self.model = self._createModel(STATE_CNT,ACTION_CNT,'linear')
        self.model_ = self._createModel(STATE_CNT,ACTION_CNT,'linear')  # target network


    def train(self, x, y, epoch=1, verbose=0):
        # x=input, y=target, batch_size = Number of samples per gradient update
        # nb_epoch = number of the epoch,
        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2
        # for one log line per epoch.
        self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01  # epsilon
    a = 0.6  # alpha
    # For PER (Prioritized Experience Replay): (error+e)^a

    def __init__(self, capacity):

        # Sumtree sucht tupel mit O(log n) anstatt O(n) mit Array
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a


    def add(self, error, sample):
        # ads new Sample to memory
        ############## so far without random thingy ###############
        #if sample[3] is not None: sample = RL_Algo.get_random_equal_state(sample)
        p = self._getPriority(error)
        self.tree.add(p, sample)
        #[self.tree.add(p, sam) for sam in samples]

    def sample(self, n):

        # computes a batch of random saved samples
        # n = amount of samples in batch

        batch = []
        segment = self.tree.total() / n

        for i in range(n):  # tree is divided into segments of equal size
            a = segment * i  # take one sample out of every segment
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)  # get with O(log n)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):  # Update Priority of a sample
        p = self._getPriority(error)
        self.tree.update(idx, p)

        #-------------------- AGENT ---------------------------


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self):

        self.brain = DQN_Brain()
        #self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        # act with epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_CNT - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):
        # observes new sample
        # sample in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        # Computes (Input, Output, Error) tuples -->Q-Learning happens here
        no_state = np.zeros(STATE_CNT)

        states = np.array([o[1][0] for o in batch])  # stores all states
        states_ = np.array([(no_state if o[1][3] is None else o[1][3])
                               for o in batch])  # stores only final states
        
        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = np.zeros((len(batch), STATE_CNT[0], STATE_CNT[1], STATE_CNT[2]))
        y = np.zeros((len(batch), ACTION_CNT))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]

            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                # double DQN (Bellmann Equation)
                t[a] = r + GAMMA * pTarget_[i][np.argmax(p_[i])]
            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):
        # Update Tuples and Errors, than train the CNN
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)


class RandomAgent:
    # Takes Random Action
    # will be used to fill memory in the beginning
    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self):
        pass

    def act(self, s):
        return random.randint(0, ACTION_CNT - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # error = reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass


#-------------------- ENVIRONMENT ---------------------


class Environment:

    def preprocess_state(self,state):

        reward = state["reward"]
        done = state["done"]
        player_map = np.zeros(shape=(STATE_CNT[1], STATE_CNT[2]))
        #print(state["playerPos"])
        player_map[(0,state["playerPos"])] = 1
        features = np.array([player_map])

        return features, reward, done

    def __init__(self):
        self.game = Learn_SinglePlayer()
        self.game.first_init()
        self.game.init( render = False)
        print(self.game.player_1.x)
        print("hii")
    def run(self, agent):

        # run one episode of the game, store the states and replay them every
        # step
        self.game.init(render = False)
        state, reward, done = self.preprocess_state(self.game.get_game_state())
        R = 0
        while True:
            # one step of game emulation
            action = agent.act(state)  # agent decides an action
            self.game.player_1.action = action - 1
            next_state, reward, done = self.preprocess_state(self.game.AI_learn_step())
            if done: # terminal state
                reward = 0
                next_state = None
            agent.observe((state, action, reward, next_state))  # agent adds the new sample

            agent.replay()
            state = next_state
            R += reward
            print("Total Reward:",R, "reward:" , reward, "action:" , self.game.player_1.action, "done" , done  )
            if done:  # terminal state
                break
        print("Total reward:", R)
        return R
#-------------------- MAIN ----------------------------

env = Environment()
agent = Agent()
randomAgent = RandomAgent()
rewards = []

try:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        env.run(randomAgent)
        #print(randomAgent.exp, "/", MEMORY_CAPACITY)

    agent.memory = randomAgent.memory

    randomAgent = None

    print("Starting learning")
    frame_count = 0
    episode_count = 0

    while True:
        if frame_count >= LEARNING_FRAMES:
            break
        episode_reward = env.run(agent)
        frame_count += episode_reward
        rewards.append(episode_reward)
        episode_count += 1

        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save the CNN
            save_counter = episode_count / SAVE_XTH_GAME
            reward_array = np.asarray(rewards)
            episodes = np.arange(0, reward_array.size, 1)
            RL_Algo.make_plot(episodes, reward_array, 'dqn')  
            RL_Algo.save_model(agent.brain.model, file = 'dqn', name = str(save_counter))

finally:
    # make plot
    reward_array = np.asarray(rewards)
    episodes = np.arange(0, reward_array.size, 1)
    
    RL_Algo.make_plot(episodes, reward_array, 'dqn')  
    RL_Algo.save_model(agent.brain.model, file = 'dqn', name = 'final')
    print("-----------Finished Process----------")
