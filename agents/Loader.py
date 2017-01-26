'''
Created on 15.01.2017

@author: Martin
'''
import game.GameMode
import pygame


if __name__ == '__main__':
  
    
    #init Game Environment
    game = game.GameMode.SinglePlayer()   
    pygame.init()    
    game.firstInit()
    game.screen = pygame.display.set_mode(game.screenSize)
    game.clock = pygame.time.Clock()
    game.init(game)
    
        """ Just For Test """
    if render:
        pygame.display.update()
        game.clock.tick(30)