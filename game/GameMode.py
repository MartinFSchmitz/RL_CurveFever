'''
Created on 01.12.2016

@author: Martin
'''

import pygame
import HumanPlayer
import AiPlayer
from Main import Main
import AI_Loader

class Learn_SinglePlayer (Main):
    
    def createPlayers(self):
        color_one = (255,100,0)
        Player = AiPlayer.AiPlayer(self.mapSize,color_one,self.screenScale)
        self.players.append(Player)
    
    def stepScore(self):
        return self.score +1
    
    def getEndScore(self):
        return self.score
    

class SinglePlayer (Main):
    
    def createPlayers(self):
        color_one = (255,100,0)
        #Player = HumanPlayer.HumanPlayer(self.mapSize,color_one,self.screenScale,"control_1")
        #Player = AiPlayer.AiPlayer(self.mapSize,color_one,self.screenScale)
        Player = AI_Loader.DQN_Player(self.mapSize,color_one,self.screenScale)
        Player.load_DQN()
        self.players.append(Player)
    
    def stepScore(self):
        return self.score +1
    
    def getEndScore(self):
        return self.score
    
class MultiPlayer (Main):
    
    def createPlayers(self):
        #self.Player = AiPlayer.AiPlayer(self.screenSize,color_one)
        color_one = (255,100,0)
        color_two = (100,255,0)
        Player_1 = HumanPlayer.HumanPlayer(self.smapSize,color_one,self.screenScale)
        Player_2 = AiPlayer.AiPlayer(self.mapSize,color_two,self.screenScale)
        
        self.players.append(Player_1)
        self.players.append(Player_2)
    
    def stepScore(self):
        return self.score
    
    def getEndScore(self):
        return self.score
    
        