'''
Created on 21.11.2016

@author: Martin
'''
import pygame
from Player import Player

class HumanPlayer(Player):
    
    
    def handle_input(self, event):
        if event.type == pygame.KEYDOWN and event.key == self.actions["right"]:
            self.rotate = 1
        if event.type == pygame.KEYUP and event.key == self.actions["right"] and self.rotate==1:
            self.rotate = 0
        if event.type == pygame.KEYDOWN and event.key == self.actions["left"]:
            self.rotate = -1
        if event.type == pygame.KEYUP and event.key == self.actions["left"] and self.rotate==-1:
            self.rotate = 0