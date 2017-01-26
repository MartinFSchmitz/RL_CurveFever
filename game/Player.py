'''
Created on 21.11.2016

@author: Martin
'''

import pygame
import math
import random
from cmath import sqrt
 
class Player(object):
    
    def __init__(self,mapSize,color,
                 screenScale,control=None):  
        
        minWalldistY = mapSize[1]/10
        minWalldistX = mapSize[0]/10
        rndy = random.randint(minWalldistY,mapSize[1] - minWalldistY )
        rndx = random.randint(minWalldistX,mapSize[0] - minWalldistX )
        
        if control == "control_1":     
            self.actions = {
                "left": pygame.K_LEFT,
                "right": pygame.K_RIGHT
                }

        elif control == "control_2":     
            self.actions = {
                "left": pygame.K_a,
                "right": pygame.K_d
                }
        
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
        self.path = [(self.x,self.y)]
        self.action = 0
        self.mapSize=mapSize

    def posUpdated(self):
        
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
        self.path.append((int(self.x),int(self.y)))
        
    def draw(self,screen):
        halfScale = self.screenScale/2
        
        x = int(self.x)
        y = int(self.y)
        for i in range(-halfScale,halfScale):
            for j in range(-halfScale,halfScale):
                
                pygame.Surface.set_at(screen,(x*self.screenScale+i,y*self.screenScale+j), self.color) 
            
    def handle_input(self, event):
        pass
    def  doAction(self, action, a = None, b= None):
        pass

        