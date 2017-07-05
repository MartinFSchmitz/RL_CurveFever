'''
Created on Feb 22, 2017

@author: marti
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import random
import math
import numpy as np
from SumTree import SumTree
import pygame
from CurveFever import Learn_SinglePlayer

from Preprocessor import LFAPreprocessor
import RL_Algo
import pickle

""" Q-Learning with linear Function Approximation and the ideas of DQN """

""" Hyperparameters """

GAMEMODE = "single"
ALGORITHM = "lfa"
PRINT_RESULTS = True

# board size
SIZE = 30
STATE_CNT = 0 # will be set later
# amount of possible actions for the agent
ACTION_CNT = 4  # left, right, straight
# capacity of memory to store experiences
MEMORY_CAPACITY = 200000  # change to 500 000 (1 000 000 in original paper)

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.0001

# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 750
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

SAVE_XTH_GAME = 10000  # all x games, save the CNN
LEARNING_FRAMES = 1000000
LEARNING_EPISODES = 50000

ALPHA = 0.005  #0.005 # dont change

game = RL_Algo.init_game(GAMEMODE, ALGORITHM)
pre = LFAPreprocessor(SIZE)
# Hack to not always set STATE_CNT manually
STATE_CNT = len(pre.lfa_preprocess_state_feat( game.get_game_state())[0])


#-------------------- BRAIN ---------------------------
""" Class that contains the Linear Function Aproximator (LFA) and the functions to use and modify it """

class Brain:

    def __init__(self):
        """ Output: 2 LFAs
            1. Q-Learning LFA
            2. Target LFA """

        # separate model for each action in the environment's action space.
        
        self.model = []
        
        for _ in range(ACTION_CNT):
            m = np.zeros(STATE_CNT)
            self.model.append(m)
            
        self.updateTargetModel()

    def train(self, x, y, a,errors, epoch=1, verbose=0):
        """ Trains the LFA with given batch of (x,y,a,error) tuples
        Perform one parameter update for whole Batch """
        state = [[], [], [],[]] # to change when actionCnt changes!
        target = [[], [], [],[]]
        batch_size = a.size # could also use any other given variable .size
        for i in range(batch_size):
            action = int(a[i])
            state[action].append(x[i])
            target[action].append(errors[i])
        for act in range(ACTION_CNT):
            if(state[act] != []):
                states = np.array(state[act])
                targets = np.hstack(np.array(target[act]))
                # formula for parameter update (gradient descent) in q-learning:
                delta = ALPHA *  np.dot(targets , states) 
                self.model[act] = self.model[act] + delta
                
    def train_small(self, x, y, a,errors, epoch=1, verbose=0):
        """ Trains the LFA with given (x,y,a, error) tuples
        Perform one parameter update for one tuple """
        act = int(a[0])
        # formula for parameter update (gradient descent) in q-learning:
        self.model[act] = self.model[act] + ALPHA * errors[0] * x[0] 

    def predict(self, s, target=False):
        
        """ Predicts Output of the LFA for given batch of input states s 
        Uses either the normal or the target LFA """

        batch_size = int(np.array(s).size / STATE_CNT)
        pred = np.zeros((batch_size, ACTION_CNT))

        if target:

            for i in range(batch_size):
                
                pred[i] = [ np.inner(m,s[i].reshape(1, -1)) for m in self.model_]
        else:

            for i in range(batch_size):
                pred[i] = [ np.inner(m,s[i].reshape(1, -1)) for m in self.model]
                
                #pred[i] = [m.predict(s[i].reshape(1, -1))[0]
                #for m in self.model]
        return pred

    def updateTargetModel(self):

        a = copy.copy(self.model)
        self.model_ = a


#-------------------- MEMORY --------------------------
"""Class to store the Experience, created by the simulation """
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01  # epsilon
    a = 0.6  # alpha
    # For PER (Prioritized Experience Replay): (error+e)^a

    def __init__(self, capacity):

        # Sumtree sucht tupel mit O(log n) anstatt O(n) mit Array
        self.tree = SumTree(capacity)

    def _getPriority(self, error):

        return (error + self.e) ** self.a

    def add(self, error, sample):  # new Sample
        """ adds new Sample to memory
            Input:
                error: TD-Error and priority of the sample
                sample: (s,a,r,s') Tuple """
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        
        """ computes a batch of random stored samples
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
            #print(idx)
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

        self.brain = Brain()
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        """ choose action to take
        with epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_CNT - 1)
        else:
            return np.argmax(self.brain.predict([s]))

    def observe(self, sample):  # in (s, a, r, s_) format
        """ observes new sample 
         sample in (s, a, r, s_) format """
         
        _,_,_,error = self._getTargets([(0, sample)])
        self.memory.add(error, sample)

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

        x = np.zeros((len(batch), STATE_CNT))
        y = np.zeros((len(batch), ACTION_CNT))
        z = np.zeros(len(batch))
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
            z[i] = a
            errors[i] =  t[a] - oldVal
            #print(errors[i])
        return (x, y, z, errors)
    def replay(self):
        """ Update Tuples and Errors, then train the LFA """
        batch = self.memory.sample(BATCH_SIZE)
        x, y, a, errors = self._getTargets(batch)
        # update errors
        abs_err = abs(errors)
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, abs_err[i])
            self.brain.train_small(x, y, a, errors)
        #self.brain.train(x, y, a, errors)


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
        self.game = RL_Algo.init_game(GAMEMODE,ALGORITHM)
        self.pre = LFAPreprocessor(SIZE+2)
        
    def run(self, agent):

        """ run one episode of the game, store the states and replay them every
         step """
        self.game.init(render = False)
        state, _,_= self.pre.lfa_preprocess_state_feat(self.game.get_game_state())  # 1st frame no action
        R = 0

        while True:
            # one step of game emulation
            action = agent.act(state)  # agent decides an action
            # converts interval (0,2) to (-1,1)
            self.game.player_1.action = action
            next_state, reward, done = self.pre.lfa_preprocess_state_feat(self.game.AI_learn_step())
            if done: # terminal state
                reward = 0
                next_state = None
            agent.observe((state, action, reward, next_state))  # agent adds the new sample
            agent.replay()
            state = next_state
            R += reward
            if done:  # terminal state
                break
        if PRINT_RESULTS: print("Total reward:", R)
        return R
#-------------------- MAIN ----------------------------

""" Run Everything, and save models afterwards
First get a full Memory with the Random Agent then use the normal Agent to to Q-Learning """
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

    # randomAgent = None

    print("-------Starting learning-------")
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

            RL_Algo.make_plot( rewards, 'lfa', 100)  
            pickle.dump(agent.brain.model, open(
                        'data/lfa/save.p', 'wb'))
    
finally:
    
    # make plot
    RL_Algo.make_plot( rewards, 'lfa',100,save_array = True)  
    pickle.dump(agent.brain.model, open(
                        'data/lfa/save.p', 'wb'))
    print("-----------Finished Process----------")
