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

from Preprocessor import CNNPreprocessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RL_Algo import Brain
import RL_Algo
import copy

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
STATE_CNT = (DEPTH, SIZE+2,SIZE+2)
ACTION_CNT = 4 # left, right, straight
 


#-------------------- BRAINS ---------------------------

class Policy_Brain():      
    
    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model,self.v_model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        K.manual_variable_initialization(False)
        #self.default_graph.finalize()    # avoid modifications
        
        self.rewards = [] # store rewards for graph

    def _build_model(self):

        l_input = Input(batch_shape = (None,STATE_CNT[0],STATE_CNT[1],STATE_CNT[2]))
        l_conv_1 = Conv2D(32, (8, 8), strides=(4,4),data_format = "channels_first", activation='relu')(l_input)
        l_conv_2 = Conv2D(64, (3, 3), data_format = "channels_first", activation='relu')(l_conv_1)
        

        l_conv_flat = Flatten()(l_conv_2)
        l_dense = Dense(units=16, activation='relu')(l_conv_flat)

        
        out_actions = Dense(units = ACTION_CNT, activation='softmax')(tf.convert_to_tensor(l_dense))

        model = Model(inputs=[l_input], outputs=[out_actions])
        model._make_predict_function()    # have to initialize before threading


#------------------------------------------------------------------

        v_l_input = Input(batch_shape = (None,STATE_CNT[0],STATE_CNT[1],STATE_CNT[2]))
        v_l_conv_1 = Conv2D(32, (8, 8), strides=(4,4),data_format = "channels_first", activation='relu')(v_l_input)
        v_l_conv_2 = Conv2D(64, (3, 3), data_format = "channels_first", activation='relu')(v_l_conv_1)
        

        v_l_conv_flat = Flatten()(v_l_conv_2)
        v_l_dense = Dense(units=16, activation='relu')(v_l_conv_flat)

        
        v_out = Dense(units = 1, activation='linear')(tf.convert_to_tensor(v_l_dense))

        v_model = Model(inputs=[v_l_input], outputs=[v_out])
        v_model._make_predict_function()    # have to initialize before threading
        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        
        v_model.compile(loss='mse', optimizer=opt)
        return model, v_model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None,STATE_CNT[0],STATE_CNT[1],STATE_CNT[2]))
        #s_t = tf.placeholder(tf.float32, shape=(None,STATE_CNT_S))
        a_t = tf.placeholder(tf.float32, shape=(None, ACTION_CNT))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted reward
        b_t = tf.placeholder(tf.float32, shape=(None, 1)) # baseline
        
        p = model(s_t) # the placeholder s_t is inserted into the model, the output will be: p,v
        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - b_t

        loss=  - log_prob * tf.stop_gradient(advantage)                                            # minimize value error

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss)

        return s_t, a_t, r_t, b_t, minimize
    

    def train(self,s,a,r,b):


        s = np.vstack([s])
        a = np.vstack(a)
        r = np.vstack(r)   
        b = np.vstack(b)
        #print("s",s,"a",a,"r",r,"b",b)
        s_t, a_t, r_t,b_t,  minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r, b_t: b})
        
    def predict(self, s):
        with self.default_graph.as_default():
            p= self.model.predict(s)
            return p
        
    def predictOne(self, s,):
        state =s.reshape(1, STATE_CNT[0], STATE_CNT[1], STATE_CNT[2])
        return self.predict(state).flatten()
    
    def train_v(self,states, target, epoch=1, verbose=0):
        #reshaped_states = states.reshape(1 ,STATE_CNT[0] , STATE_CNT[1], STATE_CNT[2])
        target = np.vstack(target)
        self.v_model.fit(states, target, batch_size=len(target), nb_epoch=epoch, verbose=verbose)
        
    def predict_v(self, s):

        #with self.default_graph.as_default():
        p = self.v_model.predict(s)
        return p


#------------------------------------------------------------------
class Agent:
    
    def __init__(self):
        
        self.policy_brain = Policy_Brain()
        #self.value_brain = Value_Brain() 

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
        for t in reversed(range(0, r.size)):
        
            running_add = running_add * GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r
  
    def replay(self, states,actions,rewards):

        total_return = self.discount_rewards(rewards)

        self.policy_brain.train_v(states, total_return) # RRRRRREEEEEEEwrite everything so that both brains are one class
        #        s_ = np.vstack([s_])
        baseline_value = self.policy_brain.predict_v(states)  

        #advantage = total_return - baseline_value
        #print("t", total_return,"a",advantage)
        self.policy_brain.train(states,actions,total_return,baseline_value) 

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
             
        states,actions,rewards = [],[],[]
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
            y = np.zeros([ACTION_CNT])
            y[action] = 1 
            actions.append(y) # grad that encourages the action that was tak
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

            RL_Algo.make_plot( rewards, 'reinforce', 100)  
            RL_Algo.save_model(agent.policy_brain.model, file = 'reinforce', name = str(save_counter))
        
finally:        
        # make plot
    reward_array = np.asarray(rewards)
    episodes = np.arange(0, reward_array.size, 1)
    RL_Algo.make_plot(episodes, 'reinforce', 100)  
    
    RL_Algo.save_model(agent.policy_brain.model, file = 'reinforce', name = 'final')
    print("-----------Finished Process----------")