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

from Preprocessor import CNNPreprocessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RL_Algo import Brain
import RL_Algo

'''
""" Stochastic Poilcy Gradients """


'''
# HYPER PARAMETERS
GAMMA = 0.99
LEARNING_FRAMES = 1000000
SAVE_XTH_GAME = 1000
SIZE = 34
DEPTH = 2
STATE_CNT = (DEPTH, SIZE+2,SIZE+2)
ACTION_CNT = 3 # left, right, straight
 


#-------------------- BRAINS ---------------------------

class Policy_Brain(Brain):      
    
    def __init__(self):
 
        self.model = self._createModel(STATE_CNT,ACTION_CNT,'softmax')
        
    def train(self, state, target, action, action_prob, epoch=1, verbose=0):
        # x=input, y=target, batch_size = Number of samples per gradient update
        #nb_epoch = number of the epoch, 
        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        #harsh grid
        action_array = [0,0,0]
        action_array[action]=target  # *action_prob evtl
        #  = -np.log(action_prob) * target 
        action_array =np.array([action_array])
        self.model.fit(state.reshape(1 ,2 , 36, 36), action_array, batch_size=1, nb_epoch=epoch, verbose=verbose)
       
    def predict(self, s):
        return self.model.predict_proba(s, verbose=0)

#------------------------------------------------------------------
class Value_Brain(Brain):

    def __init__(self):
        self.model = self._createModel(STATE_CNT,1,'linear')
        
    def train(self, state, target, epoch=1, verbose=0):
        state = state.reshape(1 ,2 , 36, 36)
        target = np.array([target])
        self.model.fit(state, target, batch_size=1, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        
        return self.model.predict(s, verbose=0)


#------------------------------------------------------------------
class Agent:
    
    def __init__(self):

        self.policy_brain = Policy_Brain()
        self.value_brain = Value_Brain() 
        
    def act(self, state):
        action_probs = self.policy_brain.predictOne(state) # create Array with action Probabilities, sum = 1
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs) # sample action from probabilities
        action_prob=action_probs[action]
        return action_prob, action

    def replay(self, episode):
            # Go through the episode and make policy updates
        for t, transition in enumerate(episode): # t is the counter, transition is one transition
            
            # The return after this timestep
            total_return = sum(GAMMA**i * t.reward for i, t in enumerate(episode[t:]))
            # Update our value estimator
            agent.value_brain.train(transition.state, total_return)
            # Calculate baseline/advantage
            baseline_value = agent.value_brain.predictOne(transition.state)   
            #baseline_value = 0     #temporary
            advantage = total_return - baseline_value
            # Update our policy estimator
            self.policy_brain.train(transition.state, advantage, transition.action, transition.action_prob) # do this for every transition, and use total return

#------------------------------------------------------------------
        
class Environment:
    
    def __init__(self):
        # init Game Environment
        self.game = Learn_SinglePlayer()
        self.game.first_init()
        self.game.init( render = False)
        
    def run(self, agent, pre):
        Transition = collections.namedtuple("Transition", ["state", "action","action_prob", "reward", "next_state", "done"])
             
        # Reset the environment and pick the first action
        self.game.init(render = False)
        state, reward, done = pre.cnn_preprocess_state(self.game.AI_learn_step())
        #state 
        episode = []
        all_rewards = 0
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_prob,action = agent.act(state)
            self.game.player_1.action = action - 1
            next_state, reward, done = pre.cnn_preprocess_state(self.game.AI_learn_step())
            #if done: # terminal state
            #    next_state = None                        
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, action_prob=action_prob, reward=reward, next_state=next_state, done=done))
            all_rewards += reward
            if done :
                break
            state = next_state            
        agent.replay(episode)
        
        print( "Total reward:", all_rewards )
        return all_rewards

#------------------------------------------------------------------

env = Environment()


# init Agents
agent = Agent()
pre = CNNPreprocessor(STATE_CNT)
rewards = []
try:
    print( "Start REINFORCE Learning process...")    
    
    frame_count = 0
    episode_count = 0
    
    while True:
        if frame_count >= LEARNING_FRAMES:
            break
        episode_reward = env.run(agent, pre)
        frame_count += episode_reward
        rewards.append(episode_reward)
        episode_count += 1
        # serialize model to JSON
        #model_json = agent.brain.model.to_json()
        # with open("model.json", "w") as json_file:
        # json_file.write(model_json)
        # serialize weights to HDF5
        if frame_count >10: break
        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save the CNN
            save_counter = episode_count / SAVE_XTH_GAME
            RL_Algo.save_model(agent.policy_brain.model, file = 'reinforce', name = str(save_counter))

        
finally:        
        # make plot
    reward_array = np.asarray(rewards)
    episodes = np.arange(0, reward_array.size, 1)
    RL_Algo.make_plot(episodes, reward_array, 'reinforce')  
    
    RL_Algo.save_model(agent.policy_brain.model, file = 'reinforce', name = 'final')
    print("-----------Finished Process----------")