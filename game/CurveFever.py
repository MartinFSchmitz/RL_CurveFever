'''
Created on 16.02.2017

@author: Martin
'''
import sys
import pygame
import Map
from Player import *

# Constants
COLOR_ONE = (255,0,0)
COLOR_TWO = (0,255,0)
COLOR_THREE = (0,0,255)
BG_COLOR = (0, 0, 0)
MAP_SIZE = (50,50)
SCREEN_SCALE = 10

class Main(object):

    def AI_learn_step(self):

        self.step(False)
        state = {  # Has to be a new method for Multiplayer + LFA
            "map": self.Map.map,
            "playerPos": (int(self.player_1.x), int(self.player_1.y)),
            "playerRot": self.player_1.rotation,
            "reward": self.step_score(),
            "done": self.done
        }
        return state


    def close_program(self):
        pygame.quit()
        sys.exit()

            
    def step(self, render=True, multi = False):
        """
            Perform one step of game emulation.
        """
        #Debugging
        debugger = 0
        if self.player_1.pos_updated() :debugger =0
        else: debugger +=1
        if debugger > 5:
            self.done    
            print("ERROR: Stuck in State, Rotation: ", self.player.rotation)
        
        # Save endstate as screenshot
        if (self.done):
            if(self.saveScreen > 0):
                self.save_screen("end")
            self.init(self, render)

        # Check if players are alive and update Map
        if (self.pause == False):
            
            if self.game_over(multi):
                
                self.get_end_score()
                self.done = True
            self.Map.update((self.player_1.x, self.player_1.y))
            if(multi): self.Map.update((self.player_2.x, self.player_2.y))

        # Check key inputs
        if render == True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.close_program()
                if event.type == pygame.QUIT:
                    self.close_program()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                if self.pause and event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.save_screen(str(self.saveScreen))
                
                self.player_1.handle_input(event)
                if(multi):self.player_2.handle_input(event)
            
            self.player_1.draw(self.screen)
            if(multi):self.player_2.draw(self.screen)

        # Update players
        if (self.pause == False):

            self.player_1.do_action(self.Map.map)
            if(multi):self.player_2.do_action(self.Map.map)
            self.player_1.update()
            if(multi):self.player_2.update()
            self.score += self.step_score()

    def save_screen(self, name):
        pygame.image.save(self.screen, "screenShots/shot_" + name + ".jpg")
        print ("Screenshot Saved")
        self.saveScreen += 1

    def game_over(self, multi):
        over = False
        if self.player_1.pos_updated() and self.Map.has_collision((self.player_1.x, self.player_1.y)): 
            over = True
        if(multi):    
            if self.player_2.pos_updated() and self.Map.has_collision((self.player_2.x, self.player_2.y)): 
                over = True
        return over

    def first_init(self):
        self.screenSize = (
            MAP_SIZE[0
                     ] * SCREEN_SCALE, MAP_SIZE[1] * SCREEN_SCALE)

        self.pause = False

    def init(self, game, render=True):

        self.saveScreen = 0
        self.done = False
        self.score = 0
        game.create_players()
        if render == True:
            self.screen.fill(BG_COLOR)
        self.Map = Map.Map(MAP_SIZE)
#------------------------------------------------------------------
        
class Learn_SinglePlayer (Main):
    
    def create_players(self):
        
        self.player_1 = Player(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        self.multi = False
        
    def step_score(self):
        return 1
    
    def get_end_score(self):
        return self.score
    

class SinglePlayer (Main):
    
    def create_players(self):

        #self.player_1 = HumanPlayer(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,"control_1")
        #self.player_1 = GreedyPlayer(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        #self.player_1 = QLFAPlayer(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        self.player_1 = DQNPlayer(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        #Player.load_DQN()
        #Player_1 = AI_Loader.LFA_Player(MAP_SIZE,color_one,SCREEN_SCALE)
        self.multi = False
        
    def step_score(self):
        return 1
    
    def get_end_score(self):
        return self.score
    
class MultiPlayer (Main):
    
    def create_players(self):
        #self.Player = AiPlayer.AiPlayer(self.screenSize,COLOR_THREE)
        self.player_1 = HumanPlayer(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,"control_1")
        #player_1 = GreedyPlayer(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        self.player_2 = GreedyPlayer(MAP_SIZE,COLOR_THREE,SCREEN_SCALE)
        self.multi = True
    
    def step_score(self):
        return self.score
    
    def get_end_score(self):
        return self.score
#------------------------------------------------------------------

if __name__ == '__main__':

    game = SinglePlayer()
    pygame.init()

    game.first_init()
    game.screen = pygame.display.set_mode(game.screenSize)
    game.clock = pygame.time.Clock()
    game.init(game)

    while True:
        game.step(multi=game.multi)
        pygame.display.update()
        game.clock.tick(30)
