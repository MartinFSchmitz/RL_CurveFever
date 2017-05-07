'''
Created on 16.02.2017

@author: Martin
'''
import sys
import pygame
import Map
from CurvePlayer import *
from TronPlayer import *

# Decide if you want to play Tron or Zatacka
GAME_TYPE = 'Tron'
#GAME_TYPE = 'Curve'

# Constants
COLOR_ONE = (255, 0, 0)
COLOR_TWO = (0, 255, 0)
COLOR_THREE = (0, 0, 255)
BG_COLOR = (0, 0, 0)
SIZE = 20 # Size of the Game Board  #min 34 with 3 conv Layers
MAP_SIZE = (SIZE, SIZE)
SCREEN_SCALE = 10

""" Main Class for playing the Game """
class Main(object):

    def get_game_state(self):
        state = {
            "map": self.Map.map,
            "playerPos": (int(self.player_1.x)+1, int(self.player_1.y)+1),
            "playerRot": self.player_1.rotation,
            "reward": self.step_score(),
            "done": self.done
        }
        return state

    def get_game_state_multiplayer(self):
        state = {
            "map": self.Map.map,
            "playerPos": (int(self.player_1.x), int(self.player_1.y)),
            "playerRot": self.player_1.rotation,
            "OpponentPos": (int(self.player_1.x), int(self.player_1.y)),
            "OpponentRot": self.player_1.rotation,
            "reward": self.step_score(),
            "done": self.done
        }
        return state

    def AI_learn_step(self):
        """ Method for the Agents to run while learning """
        self.step(False)
        return self.get_game_state()

    def close_program(self):
        pygame.quit()
        sys.exit()

    def step(self, render=True, multi=False):
        """
            Perform one step of game emulation.
        """
        self.player_1.update()
      

        # Check if players are alive and update Map
        if (self.pause == False):

            if self.game_over(multi):

                self.get_end_score()
                #print(self.score)
                self.done = True
                    # Save endstate as screenshot
            if (self.done):
                if(self.saveScreen > 0):
                    self.save_screen("end")
                if(render):
                    self.init()

            self.Map.update((self.player_1.x, self.player_1.y))
            if(multi):
                self.Map.update((self.player_2.x, self.player_2.y))

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
                if(multi):
                    self.player_2.handle_input(event)

            self.player_1.draw(self.screen)
            if(multi):
                self.player_2.draw(self.screen)

        # Update players
        if (self.pause == False):

            if(render):
                
                self.player_1.do_action(self.get_game_state())
                if(multi):
                    self.player_2.do_action(self.get_game_state())
            self.player_1.next_pos()
            if(multi):
                self.player_2.update()
            self.score += self.step_score()
            

    def save_screen(self, name):
        pygame.image.save(self.screen, "screenShots/shot_" + name + ".jpg")
        print("Screenshot Saved")
        self.saveScreen += 1

    def game_over(self, multi):
        """ checks, if game is over """
        over = False
        if self.player_1.pos_updated() and self.Map.has_collision(
                (self.player_1.x, self.player_1.y)):
            over = True
        if(multi):
            if self.player_2.pos_updated() and self.Map.has_collision(
                    (self.player_2.x, self.player_2.y)):
                over = True
        return over

    def first_init(self):
        """ run first time you start a game """
        self.screenSize = (
            MAP_SIZE[0
                     ] * SCREEN_SCALE, MAP_SIZE[1] * SCREEN_SCALE)

        self.pause = False

    def init(self, render=True):
        """ run everytime you start a game """
        self.saveScreen = 0
        self.done = False
        self.score = 0
        self.create_players()
        if render:
            self.screen.fill(BG_COLOR)
        self.Map = Map.Map(MAP_SIZE)
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

    def get_end_score(self):
        """ reward Agent gets in the end """
        return self.score

""" Gamemode to play as Singleplayer """
class SinglePlayer (Main):

    def create_players(self):

        #self.player_1 = HumanPlayer(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,"control_1")
        #self.player_1 = GreedyPlayer(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        #self.player_1 = QLFAPlayer_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        self.player_1 = LFA_REI_Player(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
        #self.player_1 = DQNPlayer_Tron(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        #self.player_1 = REINFORCEPlayer_Tron(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        #self.player_1 = A3CPlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        self.multi = False

    def step_score(self):
        """ reward Agent gets every step """
        return 1.0

    def get_end_score(self):
        return self.score

""" Gamemode to play as Multiplayer """
class MultiPlayer (Main):

    def create_players(self):
        #self.Player = AiPlayer.AiPlayer(self.screenSize,COLOR_THREE)
        self.player_1 = HumanPlayer(
            MAP_SIZE, COLOR_ONE, SCREEN_SCALE, "control_1")
        #player_1 = GreedyPlayer(MAP_SIZE,COLOR_TWO,SCREEN_SCALE)
        self.player_2 = GreedyPlayer(MAP_SIZE, COLOR_THREE, SCREEN_SCALE)
        self.multi = True

    def step_score(self):
        """ reward Agent gets every step """
        return self.score

    def get_end_score(self):
        """ reward Agent gets in the end """
        return self.score
#------------------------------------------------------------------


if __name__ == '__main__':

    self = SinglePlayer()
    pygame.init()

    self.first_init()
    self.screen = pygame.display.set_mode(self.screenSize)
    self.clock = pygame.time.Clock()
    self.init()

    while True:
        self.step(multi=self.multi)
        pygame.display.update()
        self.clock.tick(20)