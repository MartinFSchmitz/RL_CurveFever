'''
Created on 16.02.2017

@author: Martin
'''
import pygame
import math
import numpy as np
 
class Map(object):
    def __init__(self,screenSize):  
        
        self.size = (screenSize[0]+2,screenSize[1]+2)
        self.map = np.zeros(shape=self.size)
        self.zeroMap = self.map
        for x in range(0, self.size[1]):
            self.map[0,x] = 1
            self.map[-1,x] = 1
            self.map[x,0] = 1
            self.map[x,-1] = 1


    def has_collision(self,playerPos):
        if (self.map[(int(playerPos[0])+1,int(playerPos[1])+1)]==1): 
            return True
        else: 
            return False
            
    def update(self, playerPos):
        coords = (int(playerPos[0])+1,int(playerPos[1])+1)
        self.map[coords]=1
