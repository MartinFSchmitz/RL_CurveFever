# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 18:46:33 2016

@author: Martin
"""

import pygame
import HumanPlayer
import AiPlayer
import Map

def __init__(self):
    pygame.init()
    screenSize = (500, 500)
    Map = Map.Map(screenSize)
    
    screen = pygame.display.set_mode(screenSize)
    clock = pygame.time.Clock()
    
    # Player-Objekt erstellen.
    color_one = (255,100,0)
    Player = AiPlayer.AiPlayer(screenSize,color_one)
    #Player = HumanPlayer.HumanPlayer(screenSize)
    
    
    done = False

def step(self):

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True
            if event.type == pygame.QUIT:
                done = True
            
            Player.handle_input(event)
        Player.findDirection()
        # screen.fill((0, 0, 0))
        if Player.alive:
            Player.render(screen)
            if (Player.posUpdated() and Map.hasCollision((Player.x,Player.y))):
                 print("dead")
                 Player.alive = False
            Map.update((Player.x,Player.y))
            pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()

