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
import RL_Algo
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

""" Double "Deep Q -Network" with PER """

""" Hypertparameters """

SIZE = 20
DEPTH = 1
STATE_CNT = (DEPTH, SIZE + 2, SIZE + 2)
ACTION_CNT = 4  # left, right, straight

MEMORY_CAPACITY = 300000  # change to 200 000 (1 000 000 in original paper)

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1

# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 500000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

SAVE_XTH_GAME = 10000  # all x games, save the CNN
LEARNING_FRAMES = 50000000  # 50mio
LEARNING_EPISODES = 100000

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true           #Its like MSE in intervall (-1,1) and after this linear Error
    #self.test = False
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

#-------------------- BRAIN ---------------------------

""" Class that contains the Neural Network and the functions to use and modify it """
class DQN_Brain():

    def __init__(self):
        
        """ Output: 2 neural Networks
            1. Q-Network
            2. Target Network """
            
        self.model = self._createModel(STATE_CNT, ACTION_CNT, 'linear')
        self.model_ = self._createModel(
            STATE_CNT, ACTION_CNT, 'linear')  # target network

    def _createModel(self,input, output, act_fun): # Creating a CNN
        self.state_Cnt = input
        
        model = Sequential()
        # creates layer with 32 kernels with 8x8 kernel size, subsample = pooling layer
        #relu = rectified linear unit: f(x) = max(0,x), input will be 2 x Mapsize
    
        #model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(input),dim_ordering='th'))    
        #model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        #model.add(Convolution2D(64, 3, 3, activation='relu',input_shape=(input),dim_ordering='th'))
        
        model.add(Conv2D(32, (8, 8), strides=(4,4),data_format = "channels_first", activation='relu',input_shape=(input)))    
        model.add(Conv2D(64, (4, 4), strides=(2,2),data_format = "channels_first", activation='relu'))
        model.add(Conv2D(64, (3, 3), data_format = "channels_first", activation='relu'))
                  
        model.add(Flatten())
        model.add(Dense(output_dim=256, activation='relu'))
    
        model.add(Dense(output_dim=output, activation=act_fun))
    
        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        model.compile(loss='mse', optimizer=opt)
        return model
    
    def train(self, x, y, epoch=1, verbose=0):
        """ Trains the Network with given batch of (x,y) Tuples """
        # x=input, y=target, batch_size = Number of samples per gradient update
        # nb_epoch = number of the epoch,
        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2
        # for one log line per epoch.
        self.model.fit(x, y, batch_size=32, epochs=epoch, verbose=verbose)

    def predict(self, s, target=False):
        """ Predicts Output of the Neural Network for given batch of input states s 
        Uses either the normal or the target network"""
        if target:

            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target = False):
        """ Predicts Output of the Neural Network for given single input state s """
        state =s.reshape(1, self.state_Cnt[0], self.state_Cnt[1], self.state_Cnt[2])
        return self.predict(state, target).flatten()
    
    
    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


#-------------------- MEMORY --------------------------
"""Class to store the Experience, created by the simulation """
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01  # epsilon
    a = 0.6  # alpha
    # For PER (Prioritized Experience Replay): (error+e)^a

    def __init__(self, capacity):
        
        # Sumtree can search a tuple with O(log n) instead of O(n) with Array
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        """ adds new Sample to memory
        Input:
            error: TD-Error and priority of the sample
            sample: (s,a,r,s') Tuple """
            
        #if sample[3] is not None: sample = RL_Algo.get_random_equal_state(sample)
        p = self._getPriority(error)
        self.tree.add(p, sample)
        #[self.tree.add(p, sam) for sam in samples]

    def sample(self, n):

        """ computes a batch of random saved samples
        (Priorized Experience Replay)
        Input:
        n = amount of samples in batch """
        batch = []
        segment = self.tree.total() / n

        for i in range(n):  # tree is divided into segments of equal size
            a = segment * i  # take one sample out of every segment
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)  # get with O(log n)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        """ Update Priority of a sample """
        p = self._getPriority(error)
        self.tree.update(idx, p)

        #-------------------- AGENT ---------------------------
        
""" The Agent doing simulations in the environment,
 having a Brain a Memory and tries to learn """

class Agent:
    
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self):

        self.brain = DQN_Brain()
        #self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        """ choose action to take
        with epsilon-greedy policy """

        if random.random() < self.epsilon:
            return random.randint(0, ACTION_CNT - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):
        """ observes new sample 
         sample in (s, a, r, s_) format """
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + \
            (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        """ Computes TD-Error to batch of samples
        Q-Learning happens here
        Input:
            batch of (s,a,r,s') tuples
        Output:  
            (Input, Output, Error) tuples """
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
        """ Update Tuples and Errors, then train the CNN """
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)


class RandomAgent:
    """ Agent that Takes Random Action 
    is used to fill memory in the beginning """
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
""" The interface between the agent and the game environment """
class Environment:

    def __init__(self):
        self.game = RL_Algo.init_game()
        self.pre = CNNPreprocessor(STATE_CNT)

    def run(self, agent):
        """ run one episode of the game, store the states and replay them every
        step """
        self.game.init(render=False)
        state, reward, done = self.pre.cnn_preprocess_state(
            self.game.get_game_state())
        R = 0
        while True:
            # one step of game emulation
            action = agent.act(state)  # agent decides an action
            self.game.player_1.action = action
            next_state, reward, done = self.pre.cnn_preprocess_state(
                self.game.AI_learn_step())
            if done:  # terminal state
                #reward = 0
                next_state = None
            # agent adds the new sample
            agent.observe((state, action, reward, next_state))
            #[agent.replay() for _ in xrange (8)] #we make 8 steps because we have 8 new states
            agent.replay()
            state = next_state
            R += reward
            #print("frame:", "reward:" , reward, "action:" , self.game.player_1.action, "done" , done  )
            if done:  # terminal state
                break
        print("Total reward:", R)
        return R
#-------------------- MAIN ----------------------------

""" Run Everything, and save models afterwards
First get a full Memory with the Random Agent then use the normal Agent to to Q-Learning """
env = Environment()
agent = Agent()
randomAgent = RandomAgent()
rewards = []

#rewaaards = 0
#gaaames = 0
try:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        reward = env.run(randomAgent)
        #print(randomAgent.exp, "/", MEMORY_CAPACITY)
        #rewaaards += reward
        #gaaames +=1
    agent.memory = randomAgent.memory
    # print("seeeeeeeee",rewaaards/gaaames)
    randomAgent = None

    print("Starting learning")
    #frame_count = 0
    episode_count = 0

    while True:
        if episode_count >= LEARNING_EPISODES:
            break
        episode_reward = env.run(agent)
        #frame_count += episode_reward
        rewards.append(episode_reward)

        episode_count += 1
        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save the CNN

            save_counter = episode_count / SAVE_XTH_GAME

            RL_Algo.make_plot(rewards, 'dqn', 100)
            RL_Algo.save_model(
                agent.brain.model,
                file='dqn',
                name=str(save_counter))

finally:
    # make plot
    RL_Algo.make_plot(rewards, 'dqn', 100)
    RL_Algo.save_model(agent.brain.model, file='dqn', name='final')
    print("-----------Finished Process----------")
