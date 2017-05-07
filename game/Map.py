'''
Created on 16.02.2017

@author: Martin
'''
import pygame
import math
import numpy as np
 
class Map(object):
    def __init__(self,screenSize):  
        # initializes game Environment
        self.size = (screenSize[0]+2,screenSize[1]+2)
        self.map = np.zeros(shape=self.size)
        for x in range(0, self.size[1]):
            self.map[0,x] = 1
            self.map[-1,x] = 1
            self.map[x,0] = 1
            self.map[x,-1] = 1


    def has_collision(self,playerPos):
        # tests if a player has a collision with anything
        if (self.map[(int(playerPos[1]),int(playerPos[0]))]==1): 
            return True
        else: 
            return False
            
    def update(self, playerPos):
        # update the environment
        coords = (int(playerPos[1]),int(playerPos[0]))
        self.map[coords]=1
