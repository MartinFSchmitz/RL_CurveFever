'''
Created on 16.02.2017

@author: Martin
'''
import sys
import pygame
import Map
from CurvePlayer import *
from TronPlayer import *
import os
import random

# Decide if you want to play Tron or Zatacka
GAME_TYPE = 'Tron'
#GAME_TYPE = 'Curve'

# Constants
COLOR_ONE = (255, 0, 0)
COLOR_TWO = (0, 255, 0)
COLOR_THREE = (0, 0, 255)
BG_COLOR = (0, 0, 0)
SIZE = 40 # Size of the Game Board  #min 34 with 3 conv Layers
MAP_SIZE = (SIZE, SIZE)
SCREEN_SCALE = 10

""" Main Class for playing the Game """
class Main(object):

    def get_game_state(self, pl = "1"):
        if (pl == "2"): return self.get_game_state_p2()
        if (self.multi): return self.get_game_state_multiplayer()
        else: return self.get_game_state_singleplayer()
        
    def get_game_state_p2(self):
        state = {
            "map": self.Map.map,
            "playerPos": (int(self.player_2.x), int(self.player_2.y)),
            "playerRot": self.player_2.rotation,
            "opponentPos": (int(self.player_1.x), int(self.player_1.y)),
            "opponentRot": self.player_1.rotation,
            "reward": self.step_score(),
            "done": self.done
        }
        return state
    def get_game_state_singleplayer(self):
        state = {
            "map": self.Map.map,
            "playerPos": (int(self.player_1.x), int(self.player_1.y)),
            "playerRot": self.player_1.rotation,
            "reward": self.step_score(),
            "done": self.done
        }
        return state

    def get_game_state_multiplayer(self):
        state = {
            "map": self.Map.map,
            "playerPos": (self.player_1.x, self.player_1.y),
            "playerRot": self.player_1.rotation,
            "opponentPos": (int(self.player_2.x), int(self.player_2.y)),
            "opponentRot": self.player_2.rotation,
            "reward": self.step_score(),
            "done": self.done
        }
        return state

    def AI_learn_step(self):
        """ Method for the Agents to run while learning """
        self.step(render = False)
        return self.get_game_state()

    def close_program(self):
        pygame.quit()
        sys.exit()

    def step(self, render=True):
        """
            Perform one step of game emulation.
        """

        # Check key inputs
        if render:
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
                if(self.multi):
                    self.player_2.handle_input(event)

            self.player_1.draw(self.screen)
            if(self.multi):
                self.player_2.draw(self.screen)

        # Update players
        if (self.pause == False):

            if(render):
                self.player_1.do_action(self.get_game_state())
                #if(self.multi):
                    #self.player_2.do_action(self.get_game_state("2"))
            # update Player
            next_pos = self.player_1.next_pos()
            
            if(self.multi):
                self.player_2.do_action(self.get_game_state("2"))
                next_pos_2 = self.player_2.next_pos()
            else: next_pos_2 = None    
            self.score += self.step_score()

            # Check if player is still alive
            if self.game_over(self.multi,next_pos, next_pos_2):
                self.done = True
                
            self.player_1.update()
            if(self.multi):
                self.player_2.update()
            
            if (self.done ):
                if(self.saveScreen > 0):
                    self.save_screen("end")
                if(render):
                    self.init()                    
            else:    
                # update Map
                self.Map.update((int(self.player_1.x), int(self.player_1.y)))
                if(self.multi):
                    self.Map.update((int(self.player_2.x), int(self.player_2.y)))
            #print(self.Map.map[int(self.player_1.x), int(self.player_1.y)])
            
    def save_screen(self, name):
        pygame.image.save(self.screen, "screenShots/shot_" + name + ".jpg")
        print("Screenshot Saved")
        self.saveScreen += 1

    def game_over(self, multi, next_pos, next_pos_2):
        """ checks, if game is over """
        over = False
        #if self.player_1.pos_updated(): print("not upd")
        if self.player_1.pos_updated(next_pos) and self.Map.has_collision(
                next_pos):
            over = True
            self.won = False
        if(multi):
            if self.player_2.pos_updated(next_pos_2) and self.Map.has_collision(
                   next_pos_2):
                over = True
                self.won = True
            if (self.Map.has_collision(
                next_pos) and self.Map.has_collision(
                   next_pos_2)) or next_pos == next_pos_2:
                over = True
                self.won = None
        return over

    def first_init(self):
        """ run first time you start a game """
        self.epnum = 0.0
        self.totsc = 0.0
        self.screenSize = (
            MAP_SIZE[0
                     ] * SCREEN_SCALE, MAP_SIZE[1] * SCREEN_SCALE)

        self.pause = False

    def init(self, render=True):
        """ run everytime you start a game """
        self.won = False
        self.saveScreen = 0
        self.done = False
        self.score = 0
        self.create_players()
        if render:
            self.screen.fill(BG_COLOR)
        self.Map = Map.Map(MAP_SIZE)
        # put starting positions on Map
        self.Map.update((int(self.player_1.x), int(self.player_1.y)))
        if(self.multi):
            self.Map.update((int(self.player_2.x), int(self.player_2.y)))
#------------------------------------------------------------------
""" Different Game Modes: """

""" Game Mode for the AIs to use to create a player and Learn with it """
class Learn_SinglePlayer (Main):

    def create_players(self):
        if GAME_TYPE == 'Tron':
            self.player_1 = TronPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        elif GAME_TYPE == 'Curve':
            self.player_1 = CurvePlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        else:
            print("Error: False Gametype ")
        self.multi = False

    def step_score(self):
        """ reward Agent gets every step """
        return 1

""" Game Mode for the AIs to use to create a player and Learn with it """
class Learn_MultyPlayer (Main):

    def create_players(self):

        self.player_1 = TronPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        self.player_2 = GreedyPlayer_Tron(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        self.multi = True

    def step_score(self):
        """ reward Agent gets every step """
        #print(self.score)
        if self.won:
            #print(SIZE**2/2)
            return SIZE**2/2 - self.score 
        else: return 1
""" Game Mode for the AIs to use to create a player and Learn with it """

class Learn_MultyPlayer_step_2 (Main):

    def set_players(self, algorithm):
        self.name = algorithm
        self.multi = True
        
        
    def create_players(self):
        self.player_1 = TronPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        opponent = random.choice(os.listdir('data/lfa_rei/training_pool/'))
        path = 'data/lfa_rei/training_pool/' + opponent
        if (opponent == 'greedy.txt'):
            self.player_2 = GreedyPlayer_Tron(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        else:
            self.player_2 = LFA_REI_Player_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE, path = path)


    def step_score(self):
        """ reward Agent gets every step """
        if self.won:
            return 100
        elif self.done==True and self.won == False:
            return -100
        else: return 0


                
""" Gamemode to play as Singleplayer """
class SinglePlayer (Main):

    def create_players(self):

        #self.player_1 = HumanPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,"control_1")
        #self.player_1 = GreedyPlayer_Tron(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        #self.player_1 = QLFAPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        #self.player_1 = LFA_REI_Player(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        #self.player_1 = DQNPlayer_Tron(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        self.player_1 = REINFORCEPlayer_Tron(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        #self.player_1 = A3CPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        self.multi = False

    def step_score(self):
        """ reward Agent gets every step """
        return 1.0

""" Gamemode to play as Multiplayer """
class MultiPlayer (Main):

    def create_players(self):
        #self.Player = AiPlayer.AiPlayer(self.screenSize,COLOR_THREE)
        #self.player_1 = HumanPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE, "control_1")
        #player_1 = GreedyPlayer(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        self.player_1 = LFA_REI_Player_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,"m2_ohne_greedy.p" )
        self.player_2 = LFA_REI_Player_Tron(MAP_SIZE,COLOR_THREE,SCREEN_SCALE,"m2_mit_greedy.p" )
        #self.player_2 = GreedyPlayer_Tron(MAP_SIZE, COLOR_THREE, SCREEN_SCALE)
        self.multi = True

    def step_score(self):
        """ reward Agent gets every step """
        return self.score

#------------------------------------------------------------------

if __name__ == '__main__':

    self = SinglePlayer()
    pygame.init()

    self.first_init()
    self.screen = pygame.display.set_mode(self.screenSize)
    self.init()
    self.clock = pygame.time.Clock()


    while True:
        self.step()
        pygame.display.update()
        self.clock.tick(20)
