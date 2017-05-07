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

""" Player Class for CurveFever """
class CurvePlayer(object):

    def __init__(self, mapSize, color, screenScale, control=None):
        """ Initialize Player
        Input: mapSize = Size of game Board
        color = will be set as color for the player
        screenscale = will be set as scale for the view of game window
        (only used while rendering game)
        conrtol = control settiings (only used for human players)
         """
                 
        # compute random spawning position
        minWalldistY = int(mapSize[1] / 10)
        minWalldistX = int(mapSize[0] / 10)
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

        self.speed = 0.7
        self.color = color
        self.action = 1
        self.rotSpeed = 15
        self.rotation = random.randint(0, 360/self.rotSpeed) * self.rotSpeed 
        self.alive = True
        self.mapSize = mapSize
        self.init_algorithm()

    def pos_updated(self,next_pos):
        """ Returns if the player has changed his position in last step """
        #if (int(self.oldx) == int(self.x) and int(self.oldy) == int(self.y)):
        if (int(next_pos[0]) == int(self.x) and int(next_pos[1]) == int(self.y)):
            return False
        else:
            return True

    def lose(self):
        self.close()

    def next_pos(self):
        """ Computes next player position 
        returns next player position as (x,y)
        (doesn't set player position) """
        rotate = self.action - 1
        self.rotation = math.fmod(
            self.rotation + rotate * self.rotSpeed,360)
        self.xdir = math.cos( math.radians(self.rotation))
        self.ydir = math.sin( math.radians(self.rotation))
        self.oldx = self.x
        self.oldy = self.y
        x = self.x + self.xdir * self.speed
        y = self.y + self.ydir * self.speed
        #print(x,y)
        #if(x>21 or x < 0 or y>21 or y < 0):
            #print("dead")
        return (x,y)
        # self.path.append((int(self.x),int(self.y)))

    def update(self):
        """ Update player position """

        self.x += self.xdir * self.speed
        self.y += self.ydir * self.speed

    def draw(self, screen):
        """ draw new player position on the screen 
        (only used while rendering) """
        halfScale = int(self.screenScale / 2)

        x = int(self.x)
        y = int(self.y)
        for i in range(-halfScale, halfScale):
            for j in range(-halfScale, halfScale):

                pygame.Surface.set_at(
                    screen, (x * self.screenScale + i, y * self.screenScale + j), self.color)


    def handle_input(self, event):
        """ Chooses an Action considering input
        (used by Human Players only ) """
        pass

    def do_action(self, action, a=None, b=None):
        """ Chooses an action considering the action variable
        (used for the algorithms) """
        pass

    def init_algorithm(self):
        """ initialize agents in different ways """
        pass


class HumanPlayer(CurvePlayer):

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
            self.action = 2
        if event.type == pygame.KEYUP and event.key == self.actions["right"] and self.action == 2:
            self.action = 1
        if event.type == pygame.KEYDOWN and event.key == self.actions["left"]:
            self.action = 0
        if event.type == pygame.KEYUP and event.key == self.actions["left"] and self.action == 0:
            self.action = 1


class GreedyPlayer(CurvePlayer):

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


class QLFAPlayer(CurvePlayer):

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


class LFA_REI_Player(CurvePlayer):

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


class CNNPlayer(CurvePlayer):

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


class DQNPlayer(CNNPlayer):
    def get_model(self):
        return "data/dqn/model.json"

    def load_cnn(self):
        self.cnn.load_weights("data/dqn/model_final.h5")

    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        return np.argmax(values.flatten())  # argmax(Q(s,a))


class REINFORCEPlayer(CNNPlayer):
    def get_model(self):
        return "data/reinforce/model.json"

    def load_cnn(self):
        self.cnn.load_weights("data/reinforce/model_13.0.h5")

    def choose_action(self, s):
        values = self.cnn.predict(s).flatten()
        return np.random.choice(np.arange(len(values)), p=values)
        # return np.argmax(values.flatten())  # argmax(Q(s,a)


class A3CPlayer(CNNPlayer):
    def get_model(self):
        return "data/a3c/model.json"

    def load_cnn(self):
        self.cnn.load_weights("data/a3c/model_final.h5")

    def choose_action(self, s):
        values = self.cnn.predict(s)[0].flatten()              
        #eps = 0.2
        #if random.random() < eps:
        #    return random.randint(0, 2)
        #else: return np.random.choice(np.arange(len(values)), p=values)
        return np.random.choice(np.arange(len(values)), p=values)
        # return np.argmax(values.flatten())  # argmax(Q(s,a)
