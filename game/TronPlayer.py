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

""" Module that stores the Tron player parent class and several extended player class for different algorithms etc. """

class TronPlayer(object):
    """ tron player class with basic functions and variables every player needs """
    def __init__(self, mapSize, color,
                 screenScale,gamemode = None,control= "control_1", path = None, dict = None):
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
        self.init_algorithm(gamemode, path)

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
        s_color = (20,20,20) 
        x = int(self.x)
        y = int(self.y)
        # paint a few fixels on the screen to show new player position
        for i in range(-halfScale, halfScale):
            for j in range(-halfScale, halfScale):

                pygame.Surface.set_at(
                    screen, (x * self.screenScale + i - halfScale, y * self.screenScale + j - halfScale ), self.color)
                
                
        for i in range(-halfScale, halfScale):       
            # draw line in center of each player
            if self.last_action == 0: # right
                pygame.Surface.set_at(
                   screen, (x * self.screenScale + i -  self.screenScale, y * self.screenScale - halfScale), s_color)
            if self.last_action == 1: # up
                pygame.Surface.set_at(
                   screen, (x * self.screenScale - halfScale, y * self.screenScale + i), s_color)
            if self.last_action == 2: # left
                pygame.Surface.set_at(
                   screen, (x * self.screenScale + i , y * self.screenScale - halfScale), s_color)
            if self.last_action == 3: # down
                pygame.Surface.set_at(
                   screen, (x * self.screenScale- halfScale, y * self.screenScale  + i - self.screenScale),s_color)
    """ functions for children classes """
    def handle_input(self, event):
        pass

    def do_action(self, action, a=None, b=None):
        pass

    def init_algorithm(self,gamemode = None, path = None):
        pass

""" Different types of players, for human player and several algorithm players """
class HumanPlayer_Tron(TronPlayer):
    """ Human Player class,
    deciding action based on key input """
    def init_algorithm(self,gamemode = None, path = None):
        
        """ different control modes: WASD and Arrows possible as keys to move the player"""
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
                "right": pygame.K_d,
                "up": pygame.K_w,
                "down": pygame.K_s
            }

    def handle_input(self, event):
        """ deciding action based on key input """
        if event.type == pygame.KEYDOWN and event.key == self.actions["right"]:
            self.action = 0

        if event.type == pygame.KEYDOWN and event.key == self.actions["up"]:
            self.action = 1

        if event.type == pygame.KEYDOWN and event.key == self.actions["left"]:
            self.action = 2

        if event.type == pygame.KEYDOWN and event.key == self.actions["down"]:
            self.action = 3


class GreedyPlayer_Tron(TronPlayer):
    """ Not RL Greedy Player 
    following its greedy policy"""
    def init_algorithm(self,gamemode = None, path = None):
        self.agent = Greedy.Greedy()
        self.agent.init(self.mapSize[0])

    def do_action(self, state):

        action = self.agent.policy(self.action,state)
        self.action = action

""" From here Algorithm Player start:
    those are used to let a already trained algorithm play
    
    They are based on following concept:
    init_algorithm:
    load trained model from file
    
    do action:
    choose action with using their trained Function approximator
    (CNN or LFA)
 """

""" LFA based Player Classes """
class QLFAPlayer_Tron(TronPlayer):

    def init_algorithm(self,gamemode = None, path = None):
        self.prepro = LFAPreprocessor(self.mapSize[0])
        if path == None:
            path = 'data/lfa/tron_trained_30/save.p'
        with open(path, 'rb') as pickle_file:
            self.models = pickle.load(pickle_file)

    def do_action(self, game_state):
        state, _, _ = self.prepro.lfa_preprocess_state_feat(game_state)
        #a = np.array([m.predict([state])[0] for m in self.models])
        a = np.array([np.inner(m, state) for m in self.models])
        # print(np.argmax(a))
        self.action = np.argmax(a)


class LFA_REI_Player_Tron(TronPlayer):

    def init_algorithm(self,gamemode = None, path = None):
        self.prepro = LFAPreprocessor(self.mapSize[0])
        if path == None:
            path = 'data/lfa_rei/tron_trained_30/policy.p'
        with open(path, 'rb') as pickle_file:
            self.models = pickle.load(pickle_file)

    def do_action(self, game_state):
        state, _, _ = self.prepro.lfa_preprocess_state_feat(game_state)
        #a = np.array([m.predict([state])[0] for m in self.models])
        a = np.array([np.inner(m, state) for m in self.models])
        #self.action = np.argmax(a)
        e_x = np.exp(a - np.max(a))
        pred = e_x / e_x.sum()
        self.action = np.random.choice(np.arange(len(pred)), p=pred)


class CNNPlayer_Tron(TronPlayer):
    """ CNN based player classes """
    def init_algorithm(self, gamemode, path = None):
        # returns a compiled model
        # identical to the previous one
        # RMSprob is a popular adaptive learning rate method
        opt = RMSprop(lr=0.00025)
        #self.dqn=load_model('save_1.h5', custom_objects={'hubert_loss': hubert_loss,'opt': opt })
        self.stateCnt = (1, self.mapSize[0] + 2, self.mapSize[1] + 2)
        self.prepro = CNNPreprocessor(self.stateCnt, gamemode)

        # load json and create model
        json_file = open(self.get_model(), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.cnn = model_from_json(loaded_model_json)
        # load weights into new model
        self.load_cnn(path)
        self.cnn.compile(loss=self.prepro.huber_loss, optimizer=opt)
        #print("Loaded model from disk")
        
    def do_action(self, state):
        s, _, _ = self.prepro.cnn_preprocess_state(state)
        s = s.reshape(1, 1, self.mapSize[0] + 2, self.mapSize[1] + 2)
        action = self.choose_action(s)
        # action Label is in interval (0,2), but actual action is in interval
        # (-1,1)
        self.action = action


class DQNPlayer_Tron(CNNPlayer_Tron):
    def get_model(self):
        return "data/dqn/tron_trained_30/model.json"

    def load_cnn(self, path):
        if path == None:
            self.cnn.load_weights("data/dqn/tron_trained_30/model_final.h5")
        else:
            self.cnn.load_weights(path)
    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        return np.argmax(values.flatten())  # argmax(Q(s,a))


class REINFORCEPlayer_Tron(CNNPlayer_Tron):
    def get_model(self):
        return "data/reinforce/tron_trained_30/model.json"

    def load_cnn(self, path):
        if path == None:
            self.cnn.load_weights("data/reinforce/tron_trained_30/model_final.h5")
        else:
            self.cnn.load_weights(path)

    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        #return np.random.choice(np.arange(len(values)), p=values)
        return np.argmax(values.flatten())  # argmax(Q(s,a)


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
        
        
        
        
        
        
        
        
        
        
        
