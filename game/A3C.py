'''
Created on Apr 2, 2017

@author: marti
'''
# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import time
import random
import threading

from keras.models import *
from keras.layers import *
from keras import backend as K

from Preprocessor import CNNPreprocessor

import RL_Algo

#-- constants

LOADED_DATA = 'data/a3c/p.h5'
GAMEMODE = "single" # single, multi_1, multi_2
PRINT_RESULTS = True
ALGORITHM = "a3c"

SIZE = 20
DEPTH = 1
STATE_CNT = (DEPTH, SIZE + 2, SIZE + 2)

#STATE_CNT = 3
ACTION_CNT = 4

RUN_TIME = 10  # changed to 30
THREADS = 8
OPTIMIZERS = 4
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 3e-3  # eig 5e-4

LOSS_V = .5            # v loss coefficient
LOSS_ENTROPY = .01     # entropy coefficient

#---------

""" Class that contains the CNN + Tensorflow graph for policy and state value and the functions to use and modify it """
class Brain:
    train_queue = [[], [], [], [], []]    # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)    
        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        
        if LOADED_DATA != None: self.model.load_weights(LOADED_DATA)
        
        self.default_graph.finalize()    # avoid modifications
        self.rewards = []  # store rewards for graph

    def _build_model(self):
        """ build the keras CNN model vor the policy brain """
        #l_input = Input( batch_shape=(None, STATE_CNT) )
        #l_dense = Dense(16, activation='relu')(l_input)
        #l_input = Input(batch_shape = (None,STATE_CNT_S) )
        l_input = Input(
            batch_shape=(
                None,
                STATE_CNT[0],
                STATE_CNT[1],
                STATE_CNT[2]))
        l_conv_1 = Conv2D(32, (8, 8), strides=(4,4),data_format = "channels_first", activation='relu')(l_input)
        #l_conv_2 = Conv2D(64, (4, 4), strides=(2,2),data_format = "channels_first", activation='relu')(l_conv_1)
        l_conv_3 = Conv2D(64, (3, 3), data_format = "channels_first", activation='relu')(l_conv_1)
        #model.add()

        # print(l_input)
        l_conv_flat = Flatten()(l_conv_3)
        l_dense = Dense(units=16, activation='relu')(l_conv_flat)

        out_actions = Dense(
            units=ACTION_CNT,
            activation='softmax')(
            tf.convert_to_tensor(l_dense))
        out_value = Dense(units=1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()    # have to initialize before threading

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
        # not immediate, but discounted n step reward
        r_t = tf.placeholder(tf.float32, shape=(None, 1))

        # the placeholder s_t is inserted into the model, the output will be:
        # p,v
        p, v = model(s_t)

        log_prob = tf.log(
            tf.reduce_sum(
                p *
                a_t,
                axis=1,
                keep_dims=True) +
            1e-10)
        advantage = r_t - v

        # maximize policy
        loss_policy = - log_prob * tf.stop_gradient(advantage)
        # minimize value error
        loss_value = LOSS_V * tf.square(advantage)
        # maximize entropy (regularization)
        entropy = LOSS_ENTROPY * \
            tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        """ optimizer gets samples from queue """
        # only if enough samples are in the queue
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)    # yield
            return

        with self.lock_queue:
            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack([s])
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack([s_])

        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH:
            print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask    # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        """ append new batches to queue """
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        """ Predicts Output of the DQN for given batch of input states s
        Output: distribution over probabilitys to take actions 
                and State value """
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


#---------
frames = 0
#------------------------------------------------------------------
""" The Agent doing simulations in the environment,
having a Policy-Brain a Value-Brain and tries to learn """

class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []    # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        """ get current wpsilon for epsilon greedy policy """
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * \
                (self.eps_end - self.eps_start) / self.eps_steps    # linearly interpolate

    def act(self, s):
        """ choose action to take
        chooses with the probability distribution of the Policy-Brain and epsilon greedy policy """
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, ACTION_CNT - 1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(ACTION_CNT, p=p)

            return a

    def train(self, s, a, r, s_):
        """ Train the DQN with given samples of batch """
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        # turn action into one-hot representation
        a_cats = np.zeros(ACTION_CNT)
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

    # possible edge case - if an episode ends in <N steps, the computation is
    # incorrect

#------------------------------------------------------------------

""" The interface between the agent and the game environment """
class Environment(threading.Thread):
    stop_signal = False

    def __init__(
            self,
            render=False,
            eps_start=EPS_START,
            eps_end=EPS_STOP,
            eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.game = RL_Algo.init_game(GAMEMODE,ALGORITHM)
        self.pre = CNNPreprocessor(STATE_CNT)
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def runEpisode(self):

        self.game.init(render=False)
        s, _, _ = self.pre.cnn_preprocess_state(self.game.get_game_state())
        R = 0

        while True:
            time.sleep(THREAD_DELAY)  # yield
            a = self.agent.act(s)
            self.game.player_1.action = a
            s_, r, done = self.pre.cnn_preprocess_state(
                self.game.AI_learn_step())

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break
        brain.rewards.append(R)
        if PRINT_RESULTS: print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

#------------------------------------------------------------------


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


#-- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)

NONE_STATE = np.zeros(STATE_CNT)

brain = Brain()    # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
RL_Algo.make_plot(brain.rewards, ALGORITHM, 100, save_array=True)
RL_Algo.save_model(brain.model, file=ALGORITHM, name='final')
# env_test.run()
