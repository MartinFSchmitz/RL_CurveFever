'''
Created on 17.02.2017

@author: Martin
'''

import itertools
#import matplotlib
import numpy as np
import sys
import collections
import pygame
from CurveFever import Learn_SinglePlayer
from CurveFever import Learn_MultyPlayer

from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.optimizers import *
import tensorflow as tf

from Preprocessor import LFAPreprocessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RL_Algo
import copy
import pickle


""" Stochastic Poilcy Gradients with Linear Function Approximation (LFA)"""



""" HYPER PARAMETERS """
# Load already trained models to continue training:
LOADED_DATA = 'data/lfa_rei/tron_trained_30/policy.p'
LOADED_DATA_V = 'data/lfa_rei/tron_trained_30/value.p'
# Train for singleplayer or multiplayer
GAMEMODE = "multi_1" # single, multi_1, multi_2
#print episode results
PRINT_RESULTS = True
ALGORITHM = "lfa_rei"

GAMMA = 0.99
#LEARNING_FRAMES = 10000000
LEARNING_EPISODES = 50000
SAVE_XTH_GAME = 5000

# board size
SIZE = 30 + 2
# amount of possible actions for the agent
ACTION_CNT = 4 # left, right, up, down
# learning rate
ALPHA = 0.0002 #0.0002
POLICY_BATCH_TRAIN = False

# Hack to not always set STATE_CNT manually when parameter changes
game = RL_Algo.init_game(GAMEMODE, ALGORITHM)
pre = LFAPreprocessor(SIZE)
STATE_CNT = len(pre.lfa_preprocess_state_feat( game.get_game_state())[0])
print("STATE_CNT = ",STATE_CNT)

#-------------------- BRAINS ---------------------------
""" Class that contains the Linear Function Aproximator (LFA) for the Policy 
and the functions to use and modify it """

class Policy_Brain():

    def __init__(self):

        #separate model for each action in the environment's action space.
        # use previous trained model from loaded data if given
        if (LOADED_DATA == None):
            self.model = []
            for _ in range(ACTION_CNT):
                m = np.zeros(STATE_CNT) + 0.5
                self.model.append(m)
        else:
            with open(LOADED_DATA, 'rb') as pickle_file:
                self.model = pickle.load(pickle_file)

    def train(self, states, actions, total_return, baseline_value):
        """ Trains the LFA with given batch of (state,action,reward, baseline) tuples
        Perform one parameter update for whole Batch """
        # to change when actionCnt changes !
        state = [[], [], [],[]]
        target = [[], [], [],[]]
        # could also use any other given len of a variable
        batch_size = len(actions)
        pred_states_full = self.predict(states)
        # compute advantage
        advantage = total_return - baseline_value
        pred_states = np.zeros(batch_size)
        for i in range(batch_size):
            pred_states[i] = pred_states_full[int(actions[i])]
        # compute policy gradient and loss value
        log_prob = np.log(pred_states)
        loss = np.array(-log_prob) * np.array(advantage).reshape(1, -1)[0]

        for i in range(batch_size):
            action = int(actions[i])
            state[action].append(states[i])
            target[action].append(loss[i])
        # for every LFA (There is one for every Action in Actionspace)
        for act in range(ACTION_CNT):
            if(state[act] != []):
                states = np.array(state[act])
                targets = np.hstack(np.array(target[act]))
                # make parameter update
                delta = ALPHA * np.dot(targets, states)
                self.model[act] = self.model[act] + delta

    def train_small(self, states, actions, total_return, baseline_value):  # not used
        """ Trains the LFA with given (state,action,reward, baseline) tuples
        Perform one parameter update for one tuple """
        advantage = total_return - baseline_value
        # compute policy gradient and loss value
        pred_states = self.predict([states])[actions]
        log_prob = np.log(pred_states)
        loss = -log_prob * advantage

        states = np.array(states)
        targets = np.array(loss)
        # make parameter update
        delta = ALPHA * targets * states
        self.model[actions] = self.model[actions] + delta

    def predict(self, s, target=False):
        """ Predicts Output of the LFA for given batch of input states s
        Output: distribution over probabilitys to take actions """
        # sometimes scalars instead of  [a b] arrays
        batch_size = int(np.array(s).size / STATE_CNT)
        pred = np.zeros((batch_size, ACTION_CNT))

        for i in range(batch_size):
            pred[i] = [np.inner(m, s[i].reshape(1, -1)) for m in self.model]

        # Compute softmax values for each sets of scores in x.
        x = pred[0]
        e_x = np.exp(x - np.max(x))
        pred = e_x / e_x.sum()

        return pred


#------------------------------------------------------------------
class Value_Brain():
    """ Class that contains the Linear Function Aproximator (LFA) for the State Value
    and the functions to use and modify it """
    
    def __init__(self):
        """separate model for each action in the environment's """

        if (LOADED_DATA_V == None):
            self.model = np.zeros(STATE_CNT) + 0.5
        else:
            with open(LOADED_DATA_V, 'rb') as pickle_file:
                self.model = pickle.load(pickle_file)
    def train(self, states, errors, epoch=1, verbose=0):  # x, y, a,errors,
        """ Trains the LFA with given batch of (state,error) tuples
        Perform one parameter update for whole Batch """
        states = np.array(states)
        targets = np.hstack(np.array(errors))
        # paremeter update
        delta = ALPHA * np.dot(targets, states)
        self.model = self.model + delta

    def predict(self, s, target=False):
        """ Predicts Output of the LFA for given batch of input states s
        Output: Value to evaluate current state """
        # sometimes scalars instead of  [a b] arrays
        batch_size = int(np.array(s).size / STATE_CNT)
        pred = np.zeros(batch_size)
        # s[0] ist das 0te state-tupel, und pred[0] das 0te tupel von predictions
        # bei m.predict(s)[0]  braucht man die [0] um das Ergebnis, dass ein
        # array ist in ein skalar umzuwandeln

        for i in range(batch_size):
            pred[i] = np.inner(self.model, s[i].reshape(1, -1))
        pred = np.vstack(pred)
        # print(pred)
        return pred
#------------------------------------------------------------------
""" The Agent doing simulations in the environment,
having a Policy-Brain a Value-Brain and tries to learn """

class Agent:

    def __init__(self):
        """ initialize policy model and value model"""
        #K.manual_variable_initialization(True)
        self.policy_brain = Policy_Brain()
        self.value_brain = Value_Brain()
        #K.manual_variable_initialization(False)

    def act(self, state):
        """ choose action to take
        chooses with the probability distribution of the Policy-Brain"""
        # create Array with action Probabilities, sum = 1
        action_probs = self.policy_brain.predict([state])
        action = np.random.choice(
            np.arange(
                len(action_probs)),
            p=action_probs)  # sample action from probabilities
        action_prob = action_probs[action]
        return action_prob, action

    def discount_rewards(self, rewards):
        """ take 1D float array of rewards and compute discounted reward """
        r = np.vstack(rewards)
        discounted_r = np.zeros_like(r, dtype=float)
        running_add = 0.0
        r = r.flatten()
        # reverse list to compute reward with gamma^t
        for t in reversed(range(0, r.size)):

            running_add = running_add * GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def replay(self, states, actions, rewards):
        """ Train the LFA with given results of the Episode """
        # get discounted reward and baseline value
        total_return = self.discount_rewards(rewards)
        baseline_value = self.value_brain.predict(states)

        # update both: value and policy model
        self.value_brain.train(states, total_return - baseline_value)
        # use either train or small train method
        if POLICY_BATCH_TRAIN:
            self.policy_brain.train(states,actions,total_return,baseline_value)
        else:
            for i in range(len(total_return)):
                self.policy_brain.train_small(
                states[i], actions[i], total_return[i], baseline_value[i])

#------------------------------------------------------------------

""" The interface between the agent and the game environment """
class Environment:

    def __init__(self):
        """ initialize game envitonment and preprocessor for LFA """
        self.game = RL_Algo.init_game(GAMEMODE, ALGORITHM) #multiplayer?
        self.pre = LFAPreprocessor(SIZE)

    def run(self, agent):
        """ run one episode of the game, store the states and replay them every
         step """
        states, actions, rewards = [], [], []
        # Reset the environment and pick the first action
        self.game.init(render=False)
        state, reward, done = self.pre.lfa_preprocess_state_feat(
            self.game.AI_learn_step())
        all_rewards = 0

        # One step in the environment
        for t in itertools.count():

            # Take a step
            # choose action
            action_prob, action = agent.act(state)
            self.game.player_1.action = action
            # get next state tuple
            next_state, reward, done = self.pre.lfa_preprocess_state_feat(
                self.game.AI_learn_step())

            # store experience in lists
            states.append(state)
            # grad that encourages the action that was taken to be taken again
            actions.append(action)
            rewards.append(reward)

            all_rewards += reward
            # handle terminal state
            if done:
                break
            state = next_state

        states_array = np.vstack(states)
        agent.replay(states_array, actions, rewards)

        if PRINT_RESULTS: print("Total reward:", all_rewards)
        return all_rewards
#------------------------------------------------------------------

""" Run Everything, and save models afterwards """
# init Environment and Agent
env = Environment()
agent = Agent()

rewards = []
try:
    print("Starting learning")
    #frame_count = 0
    episode_count = 0
    # repeat for enough episodes:
    while True:
        #if frame_count >= LEARNING_FRAMES:
        if episode_count >= LEARNING_EPISODES:
            break
        episode_reward = env.run(agent)
        #frame_count += episode_reward
        rewards.append(episode_reward)

        episode_count += 1

        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save the LFA

            save_counter = int(episode_count / SAVE_XTH_GAME)

            RL_Algo.make_plot(rewards, 'lfa_rei', 100)
            if (GAMEMODE == "multi_2"):
                file = 'data/lfa_rei/training_pool/agent_' + str(save_counter) +'.p'  
            else:
                file = 'data/lfa_rei/policy.p'
                pickle.dump(agent.policy_brain.model, open(
                        file, 'wb'))  
                file = 'data/lfa_rei/value.p'
                pickle.dump(agent.value_brain.model, open(
                        file, 'wb'))  

finally:
    # make plot
    # save useful statistics and the trained models
    RL_Algo.make_plot(rewards, 'lfa_rei', 100, save_array=True)
    pickle.dump(agent.policy_brain.model, open(
        'data/lfa_rei/policy_final.p', 'wb'))
    pickle.dump(agent.value_brain.model, open(
        'data/lfa_rei/value_final.p', 'wb'))
    #print("mean Reward", sum(rewards)/episode_count)
    print("-----------Finished Process----------")
