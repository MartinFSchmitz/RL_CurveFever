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

from Preprocessor import CNNPreprocessor
import matplotlib
#from game.LFA_REINFORCE import LEARNING_EPISODES
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RL_Algo
import copy


""" Stochastic Poilcy Gradients - REINFORCE with Baseline """
def huber_loss(y_true, y_pred):
    """ Loss-Function that computes: sqrt(1+a^2)-1 
    Its like MSE in intervall (-1,1)
    and outside of this interval like linear Error """
    err = y_pred - y_true           
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

""" HYPER PARAMETERS """
# Load already trained models to continue training:
LOADED_DATA = None # 'data/reinforce/p.h5'
LOADED_DATA_VALUE = None # 'data/reinforce/v.h5'
# Train for singleplayer or multiplayer
GAMEMODE = "single" # single, multi_1, multi_2
#print episode results
PRINT_RESULTS = True
ALGORITHM = "reinforce"

LEARNING_RATE = 2e-4 #5e-4 #1.5e4
GAMMA = 0.99
LEARNING_FRAMES = 10000000
LEARNING_EPISODES = 50000
SAVE_XTH_GAME = 1000

#board size
SIZE = 30
#depth of input-map
DEPTH = 1
# size of parameters of state representation
STATE_CNT = (DEPTH, SIZE + 2, SIZE + 2)
# amount of possible actions for the agent
ACTION_CNT = 3  # left, right, straight
#-------------------- BRAINS ---------------------------
""" Class that contains the CNN for the Policy (containing a Keras CNN model combined with a tensorflow graph)
and the functions to use and modify it """
class Policy_Brain():

    def __init__(self):
        """ initialize tensorflow session, build the CNN and a tensorflow graph for policy network. """
        self.session = tf.Session()
        K.set_session(self.session)
        
        # build CNN
        self.model = self._build_model()
        # build graph to manually modify the cnn
        self.graph = self._build_graph(self.model)
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        
        if LOADED_DATA != None:
            self.model.load_weights(LOADED_DATA)
            
        self.rewards = []  # store rewards for graph
    
    def _build_model(self):
        """ build the keras CNN model vor the policy brain """
        # creates layers for cnn
        # CNN Layer: (amount of filters, ( Kernel Dimensions) , pooling layer size, (not importatnt param) , activation functions, given input shape for layer )
        l_input = Input(
            batch_shape=(
                None,
                STATE_CNT[0],
                STATE_CNT[1],
                STATE_CNT[2]))
        l_conv_1 = Conv2D(16, (4, 4), strides=(4,4),data_format = "channels_first", activation='relu')(l_input) #8,8 4,4 original
        l_conv_2 = Conv2D(32, (2, 2), strides=(2,2),data_format = "channels_first", activation='relu')(l_conv_1) #8,8 4,4 original
        l_conv_3 = Conv2D(32, (2, 2), data_format = "channels_first", activation='relu')(l_conv_2)

        l_conv_flat = Flatten()(l_conv_3)
        l_dense = Dense(units=16, activation='relu')(l_conv_flat)
        # make output layer with softmax function
        out_actions = Dense(
            units=ACTION_CNT,
            activation='softmax')(
            tf.convert_to_tensor(l_dense))
        # finally create model
        model = Model(inputs=[l_input], outputs=[out_actions])
        model._make_predict_function()

        return model

    def _build_graph(self, model):
        """ build the tensorflow graph to combine with keras model for the policy brain
        and compile the model """
        s_t = tf.placeholder(
            tf.float32,
            shape=(
                None,
                STATE_CNT[0],
                STATE_CNT[1],
                STATE_CNT[2]))
        #s_t = tf.placeholder(tf.float32, shape=(None,STATE_CNT_S))
        a_t = tf.placeholder(tf.float32, shape=(None, ACTION_CNT))
        # not immediate, but discounted reward
        r_t = tf.placeholder(tf.float32, shape=(None, 1))
        b_t = tf.placeholder(tf.float32, shape=(None, 1))  # baseline

        # the placeholder s_t is inserted into the model, the output will be:
        # p,v
        p = model(s_t)
        log_prob = tf.log(
            tf.reduce_sum(
                p *
                a_t,
                axis=1,
                keep_dims=True) +
            1e-10)
        advantage = r_t - b_t

        # minimize value error
        loss = -log_prob * tf.stop_gradient(advantage)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss)

        return s_t, a_t, r_t, b_t, minimize

    def train(self, s, a, r, b):

        """ Trains the LFA with given batch of (state,action,reward, baseline) tuples
        Perform one parameter update for whole Batch """
        s = np.vstack([s])
        a = np.vstack(a)
        r = np.vstack(r)
        b = np.vstack(b)
        # print("s",s,"a",a,"r",r,"b",b)
        s_t, a_t, r_t, b_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r, b_t: b})

    def predict(self, s):
        """ Predicts Output of the DQN for given batch of input states s
        Output: distribution over probabilitys to take actions """
        with self.default_graph.as_default():
            p = self.model.predict(s)
            return p

    def predictOne(self, s,):
        """ Predicts Output of the DQN for given single sample of input states s
        Output: distribution over probabilitys to take actions """
        state = s.reshape(1, STATE_CNT[0], STATE_CNT[1], STATE_CNT[2])
        return self.predict(state).flatten()


#------------------------------------------------------------------
""" class that contains the baseline (CNN for predicting the statevalue) 
and all functions to modify this"""
class Value_Brain():

    def __init__(self):
        # init state-Value prediction CNN
        self.model = self._build_model()
        # use already trained data if given
        if LOADED_DATA_VALUE != None: self.model.load_weights(LOADED_DATA_VALUE)        
    def _build_model(self):
        """ build the keras CNN model vor the value brain """
        # creates layers for cnn
        # CNN Layer: (amount of filters, ( Kernel Dimensions) , pooling layer size, (not importatnt param) , activation functions, given input shape for layer )

        l_input = Input(
            batch_shape=(
                None,
                STATE_CNT[0],
                STATE_CNT[1],
                STATE_CNT[2]))
        l_conv_1 = Conv2D(16, (4,4), strides=(4, 4), data_format="channels_first", activation='relu')(l_input)
        l_conv_2 = Conv2D(32,(2,2),data_format="channels_first",activation='relu')(l_conv_1)
        l_conv_3 = Conv2D(32,(2,2),data_format="channels_first",activation='relu')(l_conv_2)
        
        l_conv_flat = Flatten()(l_conv_3)
        l_dense = Dense(units=16, activation='relu')(l_conv_flat)

        out = Dense(
            units=1, activation='linear')(
            tf.convert_to_tensor(l_dense))

        model = Model(inputs=[l_input], outputs=[out])
        model._make_predict_function()    # have to initialize before threading
        # RMSprob is a popular adaptive learning rate method
        opt = RMSprop(lr=LEARNING_RATE)

        model.compile(loss=huber_loss, optimizer=opt)
        return model

    def train(self, states, target, epoch=1, verbose=0):
        """ Trains the DQN with given batch of (state,error) tuples
        Perform one parameter update for whole Batch """
        target = np.vstack(target)
        self.model.fit(
            states,
            target,
            batch_size=len(target),
            nb_epoch=epoch,
            verbose=verbose)

    def predict(self, s):
        """ Predicts Output of the DQN for given batch of input states s
        Output: Value to evaluate current state """
        p = self.model.predict(s)
        return p
#------------------------------------------------------------------
""" The Agent doing simulations in the environment,
having a Policy-Brain a Value-Brain and tries to learn """

class Agent:

    def __init__(self):
        #K.manual_variable_initialization(True)
        self.policy_brain = Policy_Brain()
        self.value_brain = Value_Brain()
        K.manual_variable_initialization(False)

    def act(self, state):
        """ choose action to take
        chooses with the probability distribution of the Policy-Brain"""
        # create Array with action Probabilities, sum = 1
        action_probs = self.policy_brain.predictOne(state)
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
        """ Train the CNNs with given results of the Episode """
        # compute discounted reward
        total_return = self.discount_rewards(rewards)
        # train both CNNs
        self.value_brain.train(states, total_return)
        baseline_value = self.value_brain.predict(states)
        self.policy_brain.train(states, actions, total_return, baseline_value)

#------------------------------------------------------------------

""" The interface between the agent and the game environment """
class Environment:

    def __init__(self):
        self.game = RL_Algo.init_game(GAMEMODE, ALGORITHM)
        self.pre = CNNPreprocessor(STATE_CNT, GAMEMODE)

    def run(self, agent):
        """ run one episode of the game, store the states and replay them every
        step """
        states, actions, rewards = [], [], []
        # Reset the environment and pick the first action
        self.game.init(render=False)
        state, reward, done = self.pre.cnn_preprocess_state(
            self.game.AI_learn_step())
        all_rewards = 0

        # One step in the environment
        for t in itertools.count():

            # Take a step
            
            #choose an action
            action_prob, action = agent.act(state)
            self.game.player_1.action = action
            # get nex state
            next_state, reward, done = self.pre.cnn_preprocess_state(
                self.game.AI_learn_step())

            # Keep track of the transition
            state = state.reshape(1, STATE_CNT[0], STATE_CNT[1], STATE_CNT[2])
            states.append(state)
            y = np.zeros([ACTION_CNT])
            y[action] = 1
            actions.append(y)  # gradient that encourages the action that was taken to be taken again
            rewards.append(reward)

            all_rewards += reward
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
    print("Start REINFORCE Learning process...")

    #frame_count = 0
    
    episode_count = 0

    while True:
        # staet learning, and repeat simulating episodes
        if episode_count >= LEARNING_EPISODES:
            break
        episode_reward = env.run(agent)
        if PRINT_RESULTS:  print("Episode:", episode_count)
        #frame_count += episode_reward
        rewards.append(episode_reward)
        episode_count += 1
        
        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save the CNNs and other data

            save_counter = int(episode_count / SAVE_XTH_GAME)

            RL_Algo.save_model(
                    agent.policy_brain.model,
                    file='reinforce',
                    name=str(save_counter), gamemode = GAMEMODE)
            
            RL_Algo.save_model(
                    agent.value_brain.model,
                    file='reinforce',
                    name=str(save_counter) + "_value", gamemode = GAMEMODE)
            
            RL_Algo.make_plot(rewards, 'reinforce', 100)
            
        
finally:
    # store CNN data and a reward array

    RL_Algo.make_plot(rewards, 'reinforce', 100, save_array=True)
    RL_Algo.save_model(
            agent.value_brain.model,
            file='reinforce',
            name= "final" + "_v", gamemode = "single")    
    RL_Algo.save_model(
        agent.policy_brain.model,
        file='reinforce',
        name='final')
    print("-----------Finished Process----------")
