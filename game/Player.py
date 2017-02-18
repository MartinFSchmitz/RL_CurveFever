'''
Created on 16.02.2017

@author: Martin
'''

import pygame
import math
import random
from cmath import sqrt
import numpy
import cPickle as pickle

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model
from keras.models import model_from_json

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

import Greedy
class Player(object):
    
    def __init__(self,mapSize,color,
                 screenScale,control=None):  
        
        minWalldistY = mapSize[1]/10
        minWalldistX = mapSize[0]/10
        rndy = random.randint(minWalldistY,mapSize[1] - minWalldistY )
        rndx = random.randint(minWalldistX,mapSize[0] - minWalldistX )
        
        self.control = control
        self.screenScale = screenScale
        self.oldx = rndx
        self.oldy = rndy       
        self.x = rndx
        self.y = rndy
        self.xdir = 0
        self.ydir =0
        self.rotation = random.randint(0,360)
        self.speed = 0.7
        self.color=color
        self.rotate = 0
        self.rotSpeed = 8
        self.alive = True
        #self.path = [(self.x,self.y)]
        self.action = 0
        self.mapSize=mapSize
        self.init_algorithm()

    def pos_updated(self):
        
        if (int(self.oldx) == int(self.x) and int(self.oldy) == int(self.y)): return False
        else: return True
     
    def lose (self):
        self.close()
        
    def update(self):
        self.rotation = math.fmod(self.rotation+self.rotate*self.rotSpeed,360)
        self.xdir = math.cos(math.radians(self.rotation))
        self.ydir = math.sin(math.radians(self.rotation))
        self.oldx = self.x
        self.oldy = self.y
        self.x += self.xdir*self.speed
        self.y += self.ydir*self.speed
        #self.path.append((int(self.x),int(self.y)))
        
    def draw(self,screen):
        halfScale = self.screenScale/2
        
        x = int(self.x)
        y = int(self.y)
        for i in range(-halfScale,halfScale):
            for j in range(-halfScale,halfScale):
                
                pygame.Surface.set_at(screen,(x*self.screenScale+i,y*self.screenScale+j), self.color) 
            
    def handle_input(self, event):
        pass
    def  do_action(self, action, a = None, b= None):
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
        if event.type == pygame.KEYUP and event.key == self.actions["right"] and self.rotate==1:
            self.rotate = 0
        if event.type == pygame.KEYDOWN and event.key == self.actions["left"]:
            self.rotate = -1
        if event.type == pygame.KEYUP and event.key == self.actions["left"] and self.rotate==-1:
            self.rotate = 0   
            
            
class GreedyPlayer(Player):
      
    def init_algorithm(self):
        self.agent = Greedy.Greedy()
        self.agent.init(self.mapSize[0])
        self.epsilon = 2000 # epsilon >= 3 | ToDo: try epsilon --> infinity (= no epsilon)
        
    def do_action(self, map): 
        #Fahre gerade aus, bis Abstand zu Wand < epsilon / In jedem Zeitschritt  
        action = 0
        dist_to_wall = self.agent.distance(self.rotation, map, (self.x,self.y))  
        if dist_to_wall <= self.epsilon : 
            #action = self.agent.maxdist_policy(map,  (self.x,self.y), self.rotation)
            action = self.agent.not_mindist_policy(map, (self.x,self.y), self.rotation)  
        self.rotate = action
        

class QLFAPlayer(Player):
    
    def init_algorithm(self):
        self.models = pickle.load(open('data/lfa.p', 'rb'))
        self.featurizer =  pickle.load(open('data/featurizer.p', 'rb'))
        #observation_examples.reshape(1, -1)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.featurizer.fit(self.scaler.transform((0,0,0)))
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
    
    def do_action(self, map):         
        state = (self.x,self.y,self.rotation)
        features = self.featurize_state(state)
        np.array([m.predict([features])[0] for m in self.models])
        self.action =0
        
class DQNPlayer(Player):
    
    def hubert_loss(self, y_true, y_pred):    # sqrt(1+a^2)-1
        err = y_pred - y_true           #Its like MSE in intervall (-1,1) and after this linear Error
        return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

    def init_algorithm(self):
        # returns a compiled model
        # identical to the previous one
        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        #self.dqn=load_model('save_1.h5', custom_objects={'hubert_loss': hubert_loss,'opt': opt })
        self.stateCnt = (2, self.mapSize[0]+2, self.mapSize[1]+2)

        # load json and create model
        json_file = open("data/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.dqn = model_from_json(loaded_model_json)
        # load weights into new model
        self.dqn.load_weights("data/dqn.h5")
        self.dqn.compile(loss=self.hubert_loss, optimizer=opt)

        print("Loaded model from disk")            

    def do_action(self, map): 
        diffMap = numpy.zeros(shape=(52,52))
        coords = (int(self.x),int(self.y))
        diffMap[coords] = 1
        s = numpy.array([map, diffMap]) # take map and difference map as state
        s = s.reshape(1, 2, self.mapSize[0]+2, self.mapSize[1]+2)
        action = numpy.argmax(self.dqn.predict(s)) # argmax(Q(s,a))
        
        self.rotate = action-1 # action Label is in interval (0,2), but actual action is in interval (-1,1)