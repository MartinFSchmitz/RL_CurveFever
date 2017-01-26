# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 18:46:33 2016

@author: Martin
"""

import pygame
import HumanPlayer
import AiPlayer
#import DQN_Player
import Map
import sys
import GameMode



class Main(object):
    
    def AiStep(self):

        self.step(False)
        """  state = { # Has to be a new method for Multiplayer + LFA
            "map": self.Map.map,
            "diffMap": self.Map.diffMap
        }"""
        return self.Map.map, self.Map.diffMap, 1, self.done

    def closeProgram(self): 
        pygame.quit() 
        sys.exit()
        
        
    def step(self, render = True):
        
        """
            Perform one step of game emulation.
        """

        if (self.done): self.init(self, render)
        
        for player in self.players:
            if self.game_over():
                self.getEndScore()
                self.done = True
            self.Map.update((player.x,player.y))     

 

        if render == True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.closeProgram() 
                if event.type == pygame.QUIT:
                    self.closeProgram()
                
                for player in self.players:
                    player.handle_input(event)        
            for player in self.players:    
                player.draw(self.screen)
                
        for player in self.players:
            player.doAction(player.action, self.Map.map, self.Map.diffMap)  
            player.update()
        self.stepScore()
            
    def game_over (self):
        over = False
        for player in self.players:
            if player.posUpdated() and self.Map.hasCollision((player.x,player.y)):
                over = True
        return over
    
    def firstInit(self):
        self.mapSize = (80,80)
        self.screenScale = 8
        self.screenSize = (self.mapSize[0]*self.screenScale,self.mapSize[1]*self.screenScale)
        self.BG_COLOR = (0,0,0)
        
    def init(self,game,render=True):
        
        self.done = False
        self.score = 0
        self.players = []
        game.createPlayers()
        if render==True:
            self.screen.fill(self.BG_COLOR)
        self.Map = Map.Map(self.mapSize)
         
         
      
if __name__ == '__main__':       
       
    game = GameMode.SinglePlayer()   
    pygame.init()
    
    game.firstInit()
    game.screen = pygame.display.set_mode(game.screenSize)
    game.clock = pygame.time.Clock()
    game.init(game) 
    
    while True:

        game.step()
        pygame.display.update()
        game.clock.tick(30)


