'''
Created on 16.02.2017

@author: Martin
'''

import pygame
import math
import random
from cmath import sqrt
import numpy as np
import pickle


from keras.optimizers import *
from keras.models import load_model
from keras.models import model_from_json


import Greedy
from Preprocessor import *


class TronPlayer(object):

    def __init__(self, mapSize, color,
                 screenScale, control=None):
        """ Init player and all variables
        Input:
        mapSize: size of current game board
        color: color of the player
        screenScale: (only for rendering) Scale how many pixels
         one field of the game board on the screen is
         control: (only for human player) which keys to control
        """
        # compute random spawning position
        minWalldistY = int(mapSize[1] / 10.0)
        minWalldistX = int(mapSize[0] / 10.0)
        rndy = random.randint(minWalldistY, mapSize[1] - minWalldistY)
        rndx = random.randint(minWalldistX, mapSize[0] - minWalldistX)

        # set parameters
        self.control = control
        self.screenScale = screenScale
        self.dx = 1
        self.dy = 0
        self.oldx = rndx
        self.oldy = rndy
        self.x = rndx
        self.y = rndy
        self.xdir = 0
        self.ydir = 0
        self.rotation = random.randint(0, 3) * 90
        self.speed = 1
        self.color = color
        self.last_action = 0
        self.action = 1
        self.rotSpeed = 90
        self.alive = True
        self.mapSize = mapSize
        self.init_algorithm()

    def pos_updated(self, next_pos):
        """ Did the player change its position this step? """
        return True

    def lose(self):
        self.close()

    def update(self):
        """ Update player position """

        self.x += self.dx
        self.y += self.dy

    def next_pos(self):
        """ compute next player position """
        if (self.action == 0 and self.last_action != 2):
            self.dx = 1
            self.dy = 0
            self.last_action = self.action
        elif (self.action == 1 and self.last_action != 3):
            self.dx = 0
            self.dy = -1
            self.last_action = self.action
        elif (self.action == 2 and self.last_action != 0):
            self.dx = -1
            self.dy = 0
            self.last_action = self.action
        elif (self.action == 3 and self.last_action != 1):
            self.dx = 0
            self.dy = 1
            self.last_action = self.action
        #else: print ("Value Error, Rotation is wrong")

        x = self.x + self.dx
        y = self.y + self.dy
        return (x,y)

    def draw(self, screen):
        """" render new player position on the screen """
        halfScale = int(self.screenScale / 2)

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

""" Differnt types of player, for Human Player and several Algorithm-Players """
class HumanPlayer_Tron(TronPlayer):

    def init_algorithm(self):
        if self.control == "control_1":
            self.actions = {
                "left": pygame.K_LEFT,
                "right": pygame.K_RIGHT,
                "up": pygame.K_UP,
                "down": pygame.K_DOWN
            }

        elif self.control == "control_2":
            self.actions = {
                "left": pygame.K_a,
                "right": pygame.K_d
            }

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN and event.key == self.actions["right"]:
            self.action = 0

        if event.type == pygame.KEYDOWN and event.key == self.actions["up"]:
            self.action = 1

        if event.type == pygame.KEYDOWN and event.key == self.actions["left"]:
            self.action = 2

        if event.type == pygame.KEYDOWN and event.key == self.actions["down"]:
            self.action = 3


class GreedyPlayer_Tron(TronPlayer):

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
        self.action = action


class QLFAPlayer_Tron(TronPlayer):

    def init_algorithm(self):
        self.prepro = LFAPreprocessor(self.mapSize[0])
        with open('data/lfa/save.p', 'rb') as pickle_file:
            self.models = pickle.load(pickle_file)

    def do_action(self, game_state):
        state, _, _ = self.prepro.lfa_preprocess_state_2(game_state)
        #a = np.array([m.predict([state])[0] for m in self.models])
        a = np.array([np.inner(m, state) for m in self.models])
        # print(np.argmax(a))
        self.action = np.argmax(a)


class LFA_REI_Player_Tron(TronPlayer):

    def init_algorithm(self):
        self.prepro = LFAPreprocessor(self.mapSize[0])
        with open('data/lfa_rei/save.p', 'rb') as pickle_file:
            self.models = pickle.load(pickle_file)

    def do_action(self, game_state):
        state, _, _ = self.prepro.lfa_preprocess_state_feat(game_state)
        #a = np.array([m.predict([state])[0] for m in self.models])
        a = np.array([np.inner(m, state) for m in self.models])
        self.action = np.argmax(a)

        #self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


class CNNPlayer_Tron(TronPlayer):

    def init_algorithm(self):
        # returns a compiled model
        # identical to the previous one
        # RMSprob is a popular adaptive learning rate method
        opt = RMSprop(lr=0.00025)
        #self.dqn=load_model('save_1.h5', custom_objects={'hubert_loss': hubert_loss,'opt': opt })
        self.stateCnt = (2, self.mapSize[0] + 2, self.mapSize[1] + 2)
        self.prepro = CNNPreprocessor(self.stateCnt)

        # load json and create model
        json_file = open(self.get_model(), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.cnn = model_from_json(loaded_model_json)
        # load weights into new model
        self.load_cnn()
        self.cnn.compile(loss=self.prepro.hubert_loss, optimizer=opt)
        print("Loaded model from disk")

    def do_action(self, state):
        s, _, _ = self.prepro.cnn_preprocess_state(state)
        s = s.reshape(1, 2, self.mapSize[0] + 2, self.mapSize[1] + 2)
        action = self.choose_action(s)
        # action Label is in interval (0,2), but actual action is in interval
        # (-1,1)
        self.action = action


class DQNPlayer_Tron(CNNPlayer_Tron):
    def get_model(self):
        return "data/dqn/model.json"

    def load_cnn(self):
        self.cnn.load_weights("data/dqn/model_final.h5")

    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        return np.argmax(values.flatten())  # argmax(Q(s,a))


class REINFORCEPlayer_Tron(CNNPlayer_Tron):
    def get_model(self):
        return "data/reinforce/model.json"

    def load_cnn(self):
        self.cnn.load_weights("data/reinforce/model_11.0.h5")

    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        return np.random.choice(np.arange(len(values)), p=values)
        # return np.argmax(values.flatten())  # argmax(Q(s,a)


class A3CPlayer_Tron(CNNPlayer_Tron):
    def get_model(self):
        return "data/a3c/model.json"

    def load_cnn(self):
        self.cnn.load_weights("data/a3c/model_final.h5")

    def choose_action(self, s):
        values = self.cnn.predict(s)[0].flatten()
        eps = 0.1
        if random.random() < eps:
            return random.randint(0, 2)
        else: return np.random.choice(np.arange(len(values)), p=values)
        # return np.argmax(values.flatten())  # argmax(Q(s,a)
        
        
        
        
        
        
        
        
        
        
        
