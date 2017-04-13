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

from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.optimizers import *
import tensorflow as tf

from Preprocessor import LFAPreprocessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RL_Algo import Brain
import RL_Algo
import copy
import pickle

'''
""" Stochastic Poilcy Gradients """


'''
# HYPER PARAMETERS
LEARNING_RATE = 5e-4
GAMMA = 0.99
LEARNING_FRAMES = 1000000
SAVE_XTH_GAME = 1000
SIZE = 20
DEPTH = 2
STATE_CNT = 3
ACTION_CNT = 4 # left, right, straight
ALPHA = 0.00001


#-------------------- BRAINS ---------------------------

class Policy_Brain():      
    
    def __init__(self):

        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.model = []
        
        for _ in range(ACTION_CNT):
            m = np.zeros(STATE_CNT) + 0.5
            self.model.append(m)
            
    def train(self, states,actions,total_return,baseline_value):

        state = [[], [], [], []] # to change when actionCnt changes !!!!!!!!!!!!!!!!!!!!!!
        target = [[], [], [],[]]
        batch_size = len(actions) # could also use any other given variable .size
        pred_states_full = self.predict(states)
        # ToDo  jeweils nur actionten state auswaehlen
        advantage = total_return - baseline_value
        pred_states = np.zeros(batch_size)
        for i in range(batch_size):
            pred_states[i] = pred_states_full[int(actions[i])]
        log_prob = np.log(pred_states)
        loss = np.array(-log_prob) * np.array(advantage).reshape(1, -1)[0]

        for i in range(batch_size):
            action = int(actions[i])
            state[action].append(states[i])
            target[action].append(loss[i])
        for act in range(ACTION_CNT):
            if(state[act] != []):
                states = np.array(state[act])
                targets = np.hstack(np.array(target[act]))
                delta = ALPHA *  np.dot(targets , states)
                self.model[act] = self.model[act] + delta

                
    def train_small(self, states,actions,total_return,baseline_value): # not used

        advantage = total_return - baseline_value
        pred_states = self.predict([states])[actions]
        log_prob = np.log(pred_states)
        loss = -log_prob * advantage
        
        states = np.array(states)
        targets = np.array(loss)
        delta = ALPHA *  targets * states
        self.model[actions] = self.model[actions] + delta

    def predict(self, s, target=False):
        # sometimes scalars instead of  [a b] arrays
        batch_size = int(np.array(s).size / STATE_CNT)
        pred = np.zeros((batch_size, ACTION_CNT))
        # s[0] ist das 0te state-tupel, und pred[0] das 0te tupel von predictions
        # bei m.predict(s)[0]  braucht man die [0] um das Ergebnis, dass ein
        # array ist in ein skalar umzuwandeln
        for i in range(batch_size):
            pred[i] = [ np.inner(m,s[i].reshape(1, -1)) for m in self.model]

        """Compute softmax values for each sets of scores in x."""
        x = pred[0]
        e_x = np.exp(x - np.max(x))
        pred = e_x / e_x.sum()

        return pred

    



#------------------------------------------------------------------
class Value_Brain():      
    
    def __init__(self):

        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.model = np.zeros(STATE_CNT) + 0.5
            
    def train(self, states, errors, epoch=1, verbose=0): #x, y, a,errors,
        
        states = np.array(states)
        targets = np.hstack(np.array(errors))
        delta = ALPHA *  np.dot(targets , states)
        self.model = self.model + delta
        
                
    def predict(self, s, target=False):
        # sometimes scalars instead of  [a b] arrays
        batch_size = int(np.array(s).size / STATE_CNT)
        pred = np.zeros(batch_size)
        # s[0] ist das 0te state-tupel, und pred[0] das 0te tupel von predictions
        # bei m.predict(s)[0]  braucht man die [0] um das Ergebnis, dass ein
        # array ist in ein skalar umzuwandeln

        for i in range(batch_size):
            pred[i] = np.inner(self.model,s[i].reshape(1, -1)) 
        pred = np.vstack(pred)
        #print(pred)
        return pred
#------------------------------------------------------------------
class Agent:
    
    def __init__(self):
        K.manual_variable_initialization(True)
        self.policy_brain = Policy_Brain()
        self.value_brain = Value_Brain() 
        K.manual_variable_initialization(False)
        
    def act(self, state):
        action_probs = self.policy_brain.predict([state]) # create Array with action Probabilities, sum = 1
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs) # sample action from probabilities
        action_prob=action_probs[action]
        return action_prob, action
    
    def discount_rewards(self, rewards): # so far seems to not use gamma :(
        """ take 1D float array of rewards and compute discounted reward """
        r = np.vstack(rewards)
        discounted_r = np.zeros_like(r,dtype=float)
        running_add = 0.0
        r = r.flatten()
        for t in reversed(range(0, r.size)):
        
            running_add = running_add * GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r
  
    def replay(self, states,actions,rewards):

        total_return = self.discount_rewards(rewards)
        baseline_value = self.value_brain.predict(states) 

        self.value_brain.train(states, total_return-baseline_value) # RRRRRREEEEEEEwrite everything so that both brains are one class
        #self.policy_brain.train(states,actions,total_return,baseline_value) 
        for i in range (len(total_return)):
            self.policy_brain.train_small(states[i],actions[i],total_return[i],baseline_value[i]) 

#------------------------------------------------------------------
        
class Environment:
    
    def __init__(self):
        self.game = RL_Algo.init_game()
        self.pre = LFAPreprocessor(STATE_CNT)
    
    def run(self, agent):
             
        states,actions,rewards = [],[],[]
        # Reset the environment and pick the first action
        self.game.init(render = False)
        state, reward, done = self.pre.lfa_preprocess_state_2(self.game.AI_learn_step())
        all_rewards = 0
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_prob,action = agent.act(state)
            self.game.player_1.action = action
            next_state, reward, done = self.pre.lfa_preprocess_state_2(self.game.AI_learn_step())
            #if done: # terminal state
            #    next_state = None 
                                   
            # Keep track of the transition
            #state = state.reshape(1 ,STATE_CNT[0] , STATE_CNT[1], STATE_CNT[2])
            
            states.append(state)            
            actions.append(action) # grad that encourages the action that was tak
            rewards.append(reward)

            all_rewards += reward
            if done :
                break
            state = next_state        

        states_array = np.vstack(states)
        agent.replay(states_array,actions,rewards)
        
        print( "Total reward:", all_rewards )
        return all_rewards
#------------------------------------------------------------------

env = Environment()
# init Agents
agent = Agent()

rewards = []
try:
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

            RL_Algo.make_plot( rewards, 'lfa_rei', 100)  
            pickle.dump(agent.policy_brain.model, open(
                        'data/lfa_rei/save.p', 'wb'))
    
finally:
    # make plot
    reward_array = np.asarray(rewards)
    episodes = np.arange(0, reward_array.size, 1)
    
    RL_Algo.make_plot( reward_array, 'lfa_rei',100,save_array = True)  
    pickle.dump(agent.policy_brain.model, open(
                        'data/lfa_rei/save.p', 'wb'))
    print("-----------Finished Process----------")