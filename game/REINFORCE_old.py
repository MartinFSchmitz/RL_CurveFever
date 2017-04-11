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
SIZE = 20
DEPTH = 2
STATE_CNT = (DEPTH, SIZE+2,SIZE+2)
ACTION_CNT = 4 # left, right, straight
 


#-------------------- BRAINS ---------------------------

class Policy_Brain(Brain):      
    
    def __init__(self):
 
        self.model = self._createModel(STATE_CNT,ACTION_CNT,'softmax')
        
    def train(self, states, targets, epoch=1, verbose=0):

        target_array = np.vstack(targets)
        #target_array = np.array([target])
        self.model.fit(states, target_array, batch_size=1, nb_epoch=epoch, verbose=verbose,shuffle=True)
       
    def predict(self, s, target = False):
        return self.model.predict_proba(s, verbose=0)

#------------------------------------------------------------------
class Value_Brain(Brain):

    def __init__(self):
        self.model = self._createModel(STATE_CNT,1,'linear')
        
    def train(self, states, target, epoch=1, verbose=0):
        #reshaped_states = states.reshape(1 ,STATE_CNT[0] , STATE_CNT[1], STATE_CNT[2])
        target = np.vstack(target)
        self.model.fit(states, target, batch_size=1, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target = False):
        
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
    
    def discount_rewards(self, rewards): # so far seems to not use gamma :(
        """ take 1D float array of rewards and compute discounted reward """
        r = np.vstack(rewards)
        discounted_r = np.zeros_like(r,dtype=float)
        running_add = 0.0
        r = r.flatten()
        for t in reversed(xrange(0, r.size)):
        
            running_add = running_add * GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r
  
    def replay(self, states,dlogps,rewards):

        total_return = self.discount_rewards(rewards)

        self.value_brain.train(states, total_return)
        baseline_value = agent.value_brain.predict(states)   
        advantage = total_return - baseline_value
        dlogps *= advantage
        self.policy_brain.train(states,dlogps) 

        
    def old_replay(self, episode):
            # Go through the episode and make policy updates
        for t, transition in enumerate(episode): # t is the counter, transition is one transition
            
            # The return after this timestep
            total_return = sum(GAMMA**i * t.reward for i, t in enumerate(episode[t:]))
            print(total_return)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            total_return -= np.mean(total_return)
            total_return /= np.std(total_return)
            
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
        self.game = RL_Algo.init_game()
        self.pre = CNNPreprocessor(STATE_CNT)
    def old_run(self, agent):
        Transition = collections.namedtuple("Transition", ["state", "action","action_prob", "reward", "next_state", "done"])
             
        # Reset the environment and pick the first action
        self.game.init(render = False)
        state, reward, done = self.pre.cnn_preprocess_state(self.game.AI_learn_step())
        #state 
        episode = []
        all_rewards = 0
        
        # One step in the environment
        while True:
            
            # Take a step
            action_prob,action = agent.act(state)
            self.game.player_1.action = action
            next_state, reward, done = self.pre.cnn_preprocess_state(self.game.AI_learn_step())
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
    
    def run(self, agent):
             
        states,dlogps,rewards = [],[],[]
        # Reset the environment and pick the first action
        self.game.init(render = False)
        state, reward, done = self.pre.cnn_preprocess_state(self.game.AI_learn_step())
        all_rewards = 0
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_prob,action = agent.act(state)
            self.game.player_1.action = action
            next_state, reward, done = self.pre.cnn_preprocess_state(self.game.AI_learn_step())
            #if done: # terminal state
            #    next_state = None 
                                   
            # Keep track of the transition
            state = state.reshape(1 ,STATE_CNT[0] , STATE_CNT[1], STATE_CNT[2])
            
            states.append(state)            
            # Harsh Grad ...
            y = np.zeros([ACTION_CNT])
            y[action] = -np.log(action_prob) # action_prob for subtle grid
            dlogps.append(y) # grad that encourages the action that was tak
            rewards.append(reward)

            all_rewards += reward
            if done :
                break
            state = next_state        

        states_array = np.vstack(states)
        agent.replay(states_array,dlogps,rewards)
        
        print( "Total reward:", all_rewards )
        return all_rewards
#------------------------------------------------------------------

env = Environment()
# init Agents
agent = Agent()

rewards = []
try:
    print( "Start REINFORCE Learning process...")    
    
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

            RL_Algo.make_plot( rewards, 'reinforce', 50)  
            RL_Algo.save_model(agent.policy_brain.model, file = 'reinforce', name = str(save_counter))
        
finally:        
        # make plot
    reward_array = np.asarray(rewards)
    episodes = np.arange(0, reward_array.size, 1)
    RL_Algo.make_plot(episodes, 'reinforce', 10)  
    
    RL_Algo.save_model(agent.policy_brain.model, file = 'reinforce', name = 'final')
    print("-----------Finished Process----------")