'''
Created on 17.01.2017

@author: Martin
'''

from Player import Player
import random
import numpy
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from keras.models import load_model
from keras.models import model_from_json
import sys


class Greedy_Player():

    def doAction(self, action, map, diffMap):
        pass

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true           #Its like MSE in intervall (-1,1) and after this linear Error
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )


class DQN_Player(Player):        


    def load_DQN(self):    

        # returns a compiled model
        # identical to the previous one
        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        #self.dqn=load_model('save_1.h5', custom_objects={'hubert_loss': hubert_loss,'opt': opt })
        self.stateCnt = (2, self.mapSize[0]+2, self.mapSize[1]+2)
                
                # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.dqn = model_from_json(loaded_model_json)
        # load weights into new model
        self.dqn.load_weights("dqn/model_0.h5")
        self.dqn.compile(loss=hubert_loss, optimizer=opt)

        print("Loaded model from disk")                                                 
                
        
    def doAction(self, action, map, diffMap):
        
        s = numpy.array([map, diffMap]) # take map and difference map as state
        s = s.reshape(1, 2, self.mapSize[0]+2, self.mapSize[1]+2)
        action = numpy.argmax(self.dqn.predict(s)) # argmax(Q(s,a))
        
        
        self.rotate = action-1 # action Label is in interval (0,2), but actual action is in interval (-1,1)
        print (self.rotate)

    
    def predict(self, s, target=False):

            return self.dqn.predict(s)


        