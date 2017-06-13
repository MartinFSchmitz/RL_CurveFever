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
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from spyder.utils.iofuncs import __save_array

""" Double "Deep Q -Network" with PER (DQN) """

""" Hypertparameters """
# Load already trained model to continue training:
LOADED_DATA = None #"data/dqn/trained.h5"

# Train for singleplayer or multiplayer
GAMEMODE = "single" # single, multi_1, multi_2

ALGORITHM = "dqn"

#print episode results
PRINT_RESULTS = False

#board size
SIZE = 40

#depth of input-map
DEPTH = 1

# size of parameters of state representation
STATE_CNT = (DEPTH, SIZE + 2, SIZE + 2)

# amount of possible actions for the agent
ACTION_CNT = 4  # left, right, straight

# capacity of memory to store experiences
MEMORY_CAPACITY = 30000 
# change to 200 000 (1 000 000 in original paper)

# size of mini batches for experience replay
BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1
# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 75000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

# frequencay of updating target network
UPDATE_TARGET_FREQUENCY = 10000

SAVE_XTH_GAME = 1000  # all x games, save the CNN
LEARNING_FRAMES = 50000000  # 50mio
LEARNING_EPISODES = 50000

LEARNING_RATE = 0.0004 #0.0004
#0.00025
print(LEARNING_RATE)
FRAMESKIPPING = 1

def huber_loss(y_true, y_pred):
    """ Loss-Function that computes: sqrt(1+a^2)-1 
    Its like MSE in intervall (-1,1)
    and outside of this interval like linear Error """
    err = y_pred - y_true           
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

#-------------------- BRAIN ---------------------------

""" Class that contains the Convolutional Neural Network and the functions to use and modify it
This CNN  decides the actions for the agent to take and will be trained during the algorithm"""
class DQN_Brain():

    def __init__(self):
        """ Output: 2 identical neural Networks:
            1. Q-Network
            2. Target Network """
        self.prepro = CNNPreprocessor(STATE_CNT, GAMEMODE)
        #Q-Network:
        self.model = self._createModel(STATE_CNT, ACTION_CNT, 'linear')
        # use previously trained CNN if given
        if LOADED_DATA != None: self.model.load_weights(LOADED_DATA)
        # Target-Network
        self.model_ = self._createModel(
            STATE_CNT, ACTION_CNT, 'linear')  # target network

    def _createModel(self, input, output, act_fun): # Creating a CNN
        
        """ create a CNN here 
        input = input dimension
        output = output dimension
        act_fun = activation function for the output layer"""
        
        model = Sequential()
        
        # creates layers for cnn
        # CNN Layer: (amount of filters, ( Kernel Dimensions) , pooling layer size, (not importatnt param) , activation functions, given input shape for layer  
        model.add(Conv2D(16, (4, 4), strides=(4,4),data_format = "channels_first", activation='relu',input_shape=(STATE_CNT)))    
        model.add(Conv2D(32, (2, 2), strides=(2,2),data_format = "channels_first", activation='relu'))
        model.add(Conv2D(32, (2, 2), data_format = "channels_first", activation='relu'))
        model.add(Flatten())
        model.add(Dense(activation='relu', units=256))
        model.add(Dense( activation=act_fun, units = output))
        #RMSprob is a popular adaptive learning rate method
        opt = RMSprop(lr=0.00025)
        # compile cnn with given layers, rmsprop learning method and hubert loss function 
        model.compile(loss=huber_loss, optimizer=opt)
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
        state =s.reshape(1, STATE_CNT[0], STATE_CNT[1], STATE_CNT[2])
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
        """ Adds new Sample to memory
        Input:
            error: TD-Error and priority of the sample
            sample: (s,a,r,s') Tuple """
            
        #if sample[3] is not None: sample = RL_Algo.get_random_equal_state(sample)
        p = self._getPriority(error)
        self.tree.add(p, sample)

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
 having  a Memory trying to train its Brain """

class Agent:
    
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self):
        """ initialize Agent """
        self.brain = DQN_Brain()
        #self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        """ choose action to take
        with epsilon-greedy policy """
        if GAMEMODE == "multi_2": self.epsilon = 0.1
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_CNT - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):
        """ observes new sample 
         sample in (s, a, r, s_) format """
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        # update Target model every x Episodes
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
        
        if GAMEMODE == "multi_2":
            # returns a compiled model
            # identical to the previous one
            # RMSprob is a popular adaptive learning rate method
            opt = RMSprop(lr=LEARNING_RATE)
            #self.dqn=load_model('save_1.h5', custom_objects={'hubert_loss': hubert_loss,'opt': opt })
            self.prepro = CNNPreprocessor(STATE_CNT)
    
            # load json and create model
            json_file = open(self.get_model(), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.cnn = model_from_json(loaded_model_json)
            # load weights into new model
            self.load_cnn()
            self.cnn.compile(loss=self.prepro.hubert_loss, optimizer=opt)
            print("Loaded model from disk")
     
    def do_action(self, state):
        s = state.reshape(1, 1, STATE_CNT[1], STATE_CNT[2])
        action = self.choose_action(s)
        # action Label is in interval (0,2), but actual action is in interval
        # (-1,1)
        return action
    def get_model(self):
        return "data/dqn/model.json"


    def load_cnn(self):
        self.cnn.load_weights("data/dqn/trained.h5")

    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        return np.argmax(values.flatten())  # argmax(Q(s,a))

    def act(self, s):
        if GAMEMODE == "multi_2": return self.do_action(s)
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
        self.game = RL_Algo.init_game(GAMEMODE,ALGORITHM)
        self.pre = CNNPreprocessor(STATE_CNT, GAMEMODE)

    def run(self, agent):
        """ run one episode of the game, store the states and replay them every
        step """
        self.game.init(render=False)
        state, reward, done = self.pre.cnn_preprocess_state(
            self.game.get_game_state())
        R = 0
        n = 0
        while True:
            # one step of game emulation
            #if n%FRAMESKIPPING == 0:
            action = agent.act(state)  # agent decides an action
            self.game.player_1.action = action # use action
            next_state, reward, done = self.pre.cnn_preprocess_state(
            self.game.AI_learn_step())
            if done:  # terminal state
                #reward = 0
                next_state = None
            # agent adds the new sample
            #if n%FRAMESKIPPING == 0:
            # save observes step in memory
            agent.observe((state, action, reward, next_state))
            # use experience to make a parameter update
            agent.replay()
            state = next_state
            R += reward
            n += 1
            #print("frame:", "reward:" , reward, "action:" , self.game.player_1.action, "done" , done  )
            if done:  # terminal state
                break
        if PRINT_RESULTS: print("Total reward:", R)
        return R
#-------------------- MAIN ----------------------------

""" Run Everything, and save models afterwards
First get a full Memory with the Random Agent then use the normal Agent to to Q-Learning with experience replay"""

# init everything
env = Environment()
agent = Agent()
randomAgent = RandomAgent()
rewards = []


try:
    # Use Random Agent to fill memory
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        reward = env.run(randomAgent)
        #print(randomAgent.exp, "/", MEMORY_CAPACITY)
    agent.memory = randomAgent.memory
    randomAgent = None

    print("Starting learning")
    #frame_count = 0
    episode_count = 0
    # Use DQN Agent to train its CNN
    while True:
        if episode_count >= LEARNING_EPISODES:
            break
        episode_reward = env.run(agent)
        if PRINT_RESULTS:  print("Episode:", episode_count)
        #frame_count += episode_reward
        rewards.append(episode_reward)

        episode_count += 1
        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save data

            save_counter = int(episode_count / SAVE_XTH_GAME)

            RL_Algo.make_plot(rewards, 'dqn', 100, save_array = True)
            RL_Algo.save_model(
                agent.brain.model,
                file='dqn',
                name=str(save_counter), gamemode = GAMEMODE)

finally:
    # make plot
    RL_Algo.make_plot(rewards, 'dqn', 100, save_array = True)
    RL_Algo.save_model(agent.brain.model, file='dqn', name='final')
    print("-----------Finished Process----------")
