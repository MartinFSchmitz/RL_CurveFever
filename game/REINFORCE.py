'''
Created on 11.02.2017

@author: Martin
'''

import itertools
import matplotlib
import numpy as np
import sys
import collections
import pygame
from GameMode import Learn_SinglePlayer
from keras.models import load_model
from keras.utils.np_utils import binary_logloss
from keras import optimizers


    
#def REINFORCE_loss(y_true, y_pred): 
    
    
    
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

#-------------------- BRAIN ---------------------------

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self): # Creating a CNN
        model = Sequential()

        # creates layer with 32 kernels with 8x8 kernel size, subsample = pooling layer
        #relu = rectified linear unit: f(x) = max(0,x), input will be 2 x Mapsize
        model.add(Convolution2D(64, 8, 8, subsample=(4,4), activation='relu', input_shape=(self.stateCnt)))
        #model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(self.stateCnt)))     
        #model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        #model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=512, activation='relu'))

        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        model.compile(loss=hubert_loss, optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        # x=input, y=target, batch_size = Number of samples per gradient update
        #nb_epoch = number of the epoch, 
        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        #
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        print (self.stateCnt)
        #return self.predict(s.reshape(1, (2, 80+2, 80+2)), target).flatten() #8 0 = mapsize
        return self.predict(s.reshape(1, 2, 80+2, 80+2), target).flatten() #8 0 = mapsize   try like this...
    def updateTargetModel(self):
        
        self.model_.set_weights(self.model.get_weights())

class Policy_Brain:
    
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt    
        self.model = self._createModel()


    def _createModel(self): # Creating a CNN
        model = Sequential()
        self.picked_action_prob = 1
        self.target = 1
        
        # creates layer with 32 kernels with 8x8 kernel size, subsample = pooling layer
        #relu = rectified linear unit: f(x) = max(0,x), input will be 2 x Mapsize
        model.add(Convolution2D(64, 8, 8, subsample=(4,4), activation='relu', input_shape=(self.stateCnt)))
        #model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(self.stateCnt)))     
        #model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        #model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=512, activation='relu'))

        model.add(Dense(output_dim=actionCnt, activation='softmax'))
        
        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        #REINFORCE_loss = -np.log(self.picked_action_prob) * self.target 
        model.compile(loss=self.hubert_loss, optimizer=opt)

        return model
    def REINFORCE_loss(self, y_true, y_pred):    # sqrt(1+a^2)-1
        #err = y_pred - y_true           #Its like MSE in intervall (-1,1) and after this linear Error
        a =  -np.log(self.picked_action_prob) * self.target 
        return K.mean( a, axis=-1 )    
    
    def hubert_loss(self,y_true, y_pred):    # sqrt(1+a^2)-1
        print ("true",y_true,"pred",y_pred)
        err = y_pred - y_true           #Its like MSE in intervall (-1,1) and after this linear Error
        print("result", K.mean( K.sqrt(1+K.square(err))-1, axis=-1 ))
        return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )
    
    def train(self, state, target, action, action_prob, epoch=1, verbose=0):
        # x=input, y=target, batch_size = Number of samples per gradient update
        #nb_epoch = number of the epoch, 
        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        self.picked_action_prob = action_prob
        self.target = target
        self.model.train_on_batch(state.reshape(1 ,2 , 82, 82), action, class_weight=None, sample_weight=None)
        #self.model.fit(state.reshape(1 ,2 , 82, 82), action, batch_size=1, nb_epoch=epoch, verbose=verbose)
        
    def predict(self, s):
        return self.model.predict_proba(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, 2, 80+2, 80+2)).flatten()#8 0 = mapsize   try like this...


class Value_Brain:
    def __init__(self):
        pass

    def train(self, state, total_return):
        pass

    def predict(self, state):
        return 1  

def reinforce(game, policy_brain, value_brain, num_episodes, discount_factor):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    #stats = plotting.EpisodeStats(
    #    episode_lengths=np.zeros(num_episodes),
    #    episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action","action_prob", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        game.init(game, False)
        map ,diffMap , reward, done = game.AiStep() # 1st frame no action
        state = np.array([map, diffMap])       
        
        episode = []
        all_rewards = 0 # only for debugging
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs = policy_brain.predictOne(state) # create Array with action Probabilities, sum = 1
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) # sample action from probabilities
            print(action)
            map ,diffMap, reward, done = game.AiStep()
            next_state = np.array([map, diffMap])#last two screens
            action_prob=action_probs[action]            
            #s = s_
            #next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, action_prob=action_prob, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t
            all_rewards += reward
            # Print out which step we're on, useful for debugging.

            #print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            # sys.stdout.flush()

            if t==3: #done eigentlich
                break
            state = next_state            
        print("Episode",i_episode + 1, "Reward", all_rewards )
        
        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
            # Update our value estimator
            #value_brain.train(transition.state, total_return)
            # Calculate baseline/advantage
            #baseline_value = value_brain.predictOne(transition.state)   
            baseline_value = 0     #temporary    
            advantage = total_return - baseline_value
            # Update our policy estimator
            policy_brain.train(transition.state, advantage, transition.action, transition.action_prob)
    
    stats = 0
    return stats    

#------------------------------------------------------------------
# HYPER PARAMETERS
GAMMA = 0.99
MAX_EPISODES = 2000

#------------------------------------------------------------------


# init Game Environment
game = Learn_SinglePlayer()   
game.firstInit()
#game.init(game, False)

stateCnt  = (2, game.mapSize[0]+2, game.mapSize[1]+2) # 2=Map + diffMap, height, width
actionCnt = 3 # left, right, straight
policy_brain = Policy_Brain(stateCnt, actionCnt)
value_brain = Policy_Brain(stateCnt, actionCnt)
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
print( "Start REINFORCE Learning process...")    
stats = reinforce(game, policy_brain, value_brain, MAX_EPISODES, GAMMA)