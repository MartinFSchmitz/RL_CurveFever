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

# gamemode can be either "single" or "multi"
GAMEMODE = "single"
# Constants
COLOR_ONE = (200, 20, 20)
COLOR_TWO = (20, 200, 20)
COLOR_THREE = (20, 20, 200)
BG_COLOR = (20, 20, 20)
SIZE = 30 # Size of the Game Board  #min 34 with 3 conv Layers
MAP_SIZE = (SIZE, SIZE)
# scale pixel while rendering the game
SCREEN_SCALE = 12

""" Main Class for playing the Game """

class Main(object):

    def get_game_state(self, pl = "1"):
        """ get game state for different game modes and players"""
        # decide which get_game_state method to use
        if (pl == "2"): return self.get_game_state_p2()
        if (self.multi): return self.get_game_state_multiplayer()
        else: return self.get_game_state_singleplayer()
        
    def get_game_state_p2(self):
        """ gets game state for player 2 in multiplayer mode """
        # create dictionary with informations about the game state
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
        """ get game state for single player mode """
        # create dictionary with informations about the game state
        state = {
            "map": self.Map.map,
            "playerPos": (int(self.player_1.x), int(self.player_1.y)),
            "playerRot": self.player_1.rotation,
            "reward": self.step_score(),
            "done": self.done
        }
        return state

    def get_game_state_multiplayer(self):
        """ gets game state for player 1 in multiplayer mode """
        # create dictionary with informations about the game state
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
        # compute a game stepwithout rendering
        self.step(render = False)
        return self.get_game_state()

    def close_program(self):
        """ quit the program """
        pygame.quit()
        sys.exit()

    def step(self, render=True):
        """
            Perform one step of game emulation.
        """

        # Check key inputs for human players
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
                # handle key inputs
                self.player_1.handle_input(event)
                if(self.multi):
                    self.player_2.handle_input(event)



        # Update players
        if (self.pause == False):
            # handle human player action
            if(render):
                self.player_1.do_action(self.get_game_state())
                #if(self.multi):
                    #self.player_2.do_action(self.get_game_state("2"))
            # update Player
            next_pos = self.player_1.next_pos()
            # always update possible 2nd player too
            if(self.multi):
                self.player_2.do_action(self.get_game_state("2"))
                next_pos_2 = self.player_2.next_pos()
            else: next_pos_2 = None    
            self.score += self.step_score()

            # Check if player is still alive
            if self.game_over(self.multi,next_pos, next_pos_2):
                self.done = True
            # update players
            self.player_1.update()
            if(self.multi):
                self.player_2.update()
                
            if(render):    
                self.player_1.draw(self.screen)
                if(self.multi):
                    self.player_2.draw(self.screen)
                
                
                
            # handle terminal state
            if (self.done ):
                if(self.saveScreen > 0):
                    self.save_screen("end")
                if(render):
                    self.init()                    
            # handle not-terminal state
            else:    
                # update Map
                self.Map.update((int(self.player_1.x), int(self.player_1.y)))
                if(self.multi):
                    self.Map.update((int(self.player_2.x), int(self.player_2.y)))
            #print(self.Map.map[int(self.player_1.x), int(self.player_1.y)])
            
    def save_screen(self, name):
        """ function to make screenshots """
        pygame.image.save(self.screen, "screenShots/shot_" + name + ".jpg")
        print("Screenshot Saved")
        self.saveScreen += 1

    def game_over(self, multi, next_pos, next_pos_2):
        """ checks, if game is over """
        over = False
        # does player 1 have a collision with the environment?
        if self.player_1.pos_updated(next_pos) and self.Map.has_collision(
                next_pos):
            # update winning stats and game is over
            self.stats_p2w += 1
            over = True
            self.won = False
        if(multi): # if multi, player 2 possibly had a collision
            # does player 2 have a collision with the environment?
            if self.player_2.pos_updated(next_pos_2) and self.Map.has_collision(
                   next_pos_2):
                # update winning stats and game is over
                self.stats_p1w += 1
                over = True
                self.won = True
            # do both players have a collision and the game is a draw?
            if (self.Map.has_collision(
                next_pos) and self.Map.has_collision(
                   next_pos_2)) or next_pos == next_pos_2:
                # hacky way to update stats in the right way
                if over:
                    self.stats_p1w -= 1
                    self.stats_p2w -= 1
                over = True
                self.won = None
                # keep track of winning statistics

                self.stats_draw += 1
        return over

    def first_init(self):
        """ run first time you start a game """
        self.epnum = 0.0
        self.totsc = 0.0
        self.screenSize = (
            MAP_SIZE[0
                     ] * SCREEN_SCALE, MAP_SIZE[1] * SCREEN_SCALE)

        self.pause = False
        
        # make statistics about who won how many times
        self.stats_p1w = 0
        self.stats_p2w = 0
        self.stats_draw = 0

    def init(self, render=True):
        """ run everytime you start a game """
        # initialize all variables
        # bool if player_1 won, lose or game ended in draw
        self.won = False
        # write game score
        if GAMEMODE == "multi" and render == True:
            print( "Player_1 vs. Player_2  " + str(self.stats_p1w) + " : " + str(self.stats_p2w), "draws: " + str(self.stats_draw) )
        # counter to properly name screenshots
        self.saveScreen = 0
        # bool that shows if game is over
        self.done = False
        # reset game score
        self.score = 0
        # create different players
        self.create_players()
        # clean screen if render
        if render:
            self.screen.fill(BG_COLOR)
        self.Map = Map.Map(MAP_SIZE)
        # put starting positions on Map
        self.Map.update((int(self.player_1.x), int(self.player_1.y)))
        if(self.multi):
            self.Map.update((int(self.player_2.x), int(self.player_2.y)))
#------------------------------------------------------------------
""" Different Game Modes: """


class Learn_SinglePlayer (Main):
    """ Game Mode for the AIs to use to create a player and Learn with it """
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


class Learn_MultyPlayer (Main):
    """ Game Mode for the AIs to use to create a player and Learn with it """
    def create_players(self):

        self.player_1 = TronPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        self.player_2 = GreedyPlayer_Tron(MAP_SIZE,COLOR_TWO,SCREEN_SCALE,gamemode = "multi_1")
        self.multi = True

    def step_score(self):
        """ reward Agent gets every step """
        #print(self.score)
        if self.won:
            #print(SIZE**2/2)
            return SIZE**2/2 - self.score 
        else: return 1


class Learn_MultyPlayer_step_2 (Main):
    """ Game Mode for the AIs to use to create a player and Learn with it """
    def set_players(self, algorithm):
        self.name = algorithm
        self.multi = True
        self.path = None
      
    def create_players(self):
        """ method to create players in multi_2 training step
        neeeded to spawn random 2nd polayer out of training pool
        old versions of the trained FA will be added to this training pool"""
        self.player_1 = TronPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE, gamemode = "multi_2")
        opponent = random.choice(os.listdir('data/' + self.name + '/training_pool/'))
        path = 'data/' + self.name + '/training_pool/' + opponent

        if (opponent == 'greedy.txt'):
            self.player_2 = GreedyPlayer_Tron(MAP_SIZE,COLOR_TWO,SCREEN_SCALE,gamemode = "multi_2")
        elif (self.name == "lfa_rei"):
            self.player_2 = LFA_REI_Player_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,gamemode = "multi_2", path = path)
        elif (self.name == "lfa"):
            self.player_2 = QLFAPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,gamemode = "multi_2", path = path)
        elif (self.name == "reinforce"):
            self.player_2 = REINFORCEPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,gamemode = "multi_2", path = path)
        elif (self.name == "dqn"):
            self.player_2 = DQNPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,gamemode = "multi_2", path = path)
        elif (self.name == "a3c"):
            self.player_2 = A3CPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,gamemode = "multi_2", path = path)
        else: print("error: False algorithm name")
    def step_score(self):
        """ reward Agent gets every step """
        if self.won:
            return 100
        elif self.done==True and self.won == False:
            return -100
        else: return 0


                

class SinglePlayer (Main):
    """ Gamemode to play as Singleplayer """
    def create_players(self):
        
        self.player_1 = HumanPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,"control_1")
        #self.player_1 = GreedyPlayer_Tron(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        #self.player_1 = QLFAPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        #self.player_1 = LFA_REI_Player(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        #self.player_1 = DQNPlayer_Tron(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        #self.player_1 = REINFORCEPlayer_Tron(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        #self.player_1 = A3CPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        self.multi = False

    def step_score(self):
        """ reward Agent gets every step """
        return 1.0


class MultiPlayer (Main):
    """ Gamemode to play as Multiplayer """
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
    
    if GAMEMODE == "multi":
        self = MultiPlayer()
    elif GAMEMODE == "single":
        self = SinglePlayer()
    else: print("Error: Wrong Gametype")
    pygame.init()

    self.first_init()
    self.screen = pygame.display.set_mode(self.screenSize)
    self.init()
    self.clock = pygame.time.Clock()


    while True:
        self.step()
        pygame.display.update()
        self.clock.tick(20)
