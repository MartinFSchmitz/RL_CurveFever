'''
Created on 16.02.2017

@author: Martin
'''

import pygame
import math
import random
from cmath import sqrt
import numpy as np
import cPickle as pickle

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model
from keras.models import model_from_json
from sklearn.externals import joblib

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

import Greedy
from Preprocessor import Preprocessor


class Player(object):

    def __init__(self, mapSize, color,
                 screenScale, control=None):
        # compute random spawning position
        minWalldistY = mapSize[1] / 10
        minWalldistX = mapSize[0] / 10
        rndy = random.randint(minWalldistY, mapSize[1] - minWalldistY)
        rndx = random.randint(minWalldistX, mapSize[0] - minWalldistX)

        # set parameters
        self.control = control
        self.screenScale = screenScale
        self.oldx = rndx
        self.oldy = rndy
        self.x = rndx
        self.y = rndy
        self.xdir = 0
        self.ydir = 0
        self.rotation = random.randint(0, 360)
        self.speed = 0.7
        self.color = color
        self.rotate = 0
        self.rotSpeed = 8
        self.alive = True
        self.mapSize = mapSize
        self.init_algorithm()

    def pos_updated(self):
        # Did the player change its position this step?
        if (int(self.oldx) == int(self.x) and int(self.oldy) == int(self.y)):
            return False
        else:
            return True

    def lose(self):
        self.close()

    def update(self):
        # Update player position
        self.rotation = math.fmod(
            self.rotation + self.rotate * self.rotSpeed, 360)
        self.xdir = math.cos(math.radians(self.rotation))
        self.ydir = math.sin(math.radians(self.rotation))
        self.oldx = self.x
        self.oldy = self.y
        self.x += self.xdir * self.speed
        self.y += self.ydir * self.speed
        # self.path.append((int(self.x),int(self.y)))

    def draw(self, screen):
        # render new player position on the screen
        halfScale = self.screenScale / 2

        x = int(self.x)
        y = int(self.y)
        for i in range(-halfScale, halfScale):
            for j in range(-halfScale, halfScale):

                pygame.Surface.set_at(
                    screen, (x * self.screenScale + i, y * self.screenScale + j), self.color)


    def handle_input(self, event):
        pass

    def do_action(self, action, a=None, b=None):
        pass

    def init_algorithm(self):
        pass


class HumanPlayer(Player):

    def init_algorithm(self):
        if self.control == "control_1":
            self.actions = {
                "left": pygame.K_LEFT,
                "right": pygame.K_RIGHT
            }

        elif self.control == "control_2":
            self.actions = {
                "left": pygame.K_a,
                "right": pygame.K_d
            }

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN and event.key == self.actions["right"]:
            self.rotate = 1
        if event.type == pygame.KEYUP and event.key == self.actions["right"] and self.rotate == 1:
            self.rotate = 0
        if event.type == pygame.KEYDOWN and event.key == self.actions["left"]:
            self.rotate = -1
        if event.type == pygame.KEYUP and event.key == self.actions["left"] and self.rotate == -1:
            self.rotate = 0


class GreedyPlayer(Player):

    def init_algorithm(self):
        self.agent = Greedy.Greedy()
        self.agent.init(self.mapSize[0])
        # epsilon >= 3 | ToDo: try epsilon --> infinity (= no epsilon)
        self.epsilon = 2000

    def do_action(self, map):
        # In every Timestep: drive straight until distance to Wall is < epsilon
        action = 0
        dist_to_wall = self.agent.distance(
            self.rotation, map, (self.x, self.y))
        if dist_to_wall <= self.epsilon:
            #action = self.agent.maxdist_policy(map,  (self.x,self.y), self.rotation)
            action = self.agent.not_mindist_policy(
                map, (self.x, self.y), self.rotation)
        self.rotate = action


class QLFAPlayer(Player):

    def init_algorithm(self):
        self.prepro = Preprocessor()
        self.models = joblib.load('data/lfa/model_1.pkl') 
        self.prepro.lfa_constant(self.mapSize[0])
        
    def do_action(self, game_state):    
        state, _, _ = self.prepro.lfa_preprocess_state(game_state)
        a = np.array([m.predict([state])[0] for m in self.models])
        print(a)
        self.rotate = np.argmax(a) - 1


class CNNPlayer(Player):
    
    def init_algorithm(self):
        # returns a compiled model
        # identical to the previous one
        # RMSprob is a popular adaptive learning rate method
        opt = RMSprop(lr=0.00025)
        self.prepro =Preprocessor()
        #self.dqn=load_model('save_1.h5', custom_objects={'hubert_loss': hubert_loss,'opt': opt })
        self.stateCnt = (2, self.mapSize[0] + 2, self.mapSize[1] + 2)

        # load json and create model
        json_file = open("data/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.cnn = model_from_json(loaded_model_json)
        # load weights into new model
        self.load_cnn()
        self.cnn.compile(loss=self.prepro.hubert_loss, optimizer=opt)
        print("Loaded model from disk")
        
    def do_action(self, state):
        s,_,_= self.prepro.cnn_preprocess_state(state,self.stateCnt)
        s = s.reshape(1,2, self.mapSize[0] + 2, self.mapSize[1] + 2)
        action = self.choose_action(s)
        
        # action Label is in interval (0,2), but actual action is in interval
        # (-1,1)
        self.rotate = action - 1
class DQNPlayer(CNNPlayer):
    
    def load_cnn(self):
        self.cnn.load_weights("data/dqn/model_end.h5")      
    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        return np.argmax(values.flatten())  # argmax(Q(s,a))
class REINFORCEPlayer(CNNPlayer):
    
    def load_cnn(self):
        self.cnn.load_weights("data/reinforce/model_1.h5")      
    def choose_action(self, s):
        action_probs = self.cnn.predict_proba(s).flatten()
        return np.random.choice(np.arange(len(action_probs)), p=action_probs) # sample action from probabilities
        