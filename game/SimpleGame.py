'''
Created on 03.10.2017

@author: Martin
'''
import sys
import pygame


# Constants
COLOR_ONE = (255, 0, 0)
BG_COLOR = (0, 0, 0)
SIZE = 10
MAP_SIZE = (SIZE,1) 
SCREEN_SCALE = 20


class Main(object):
    
    def get_game_state(self):

        state = {
            "playerPos": self.player_1.x,
            "reward": self.step_score(),
            "done": self.done
        }
        return state

    def AI_learn_step(self):
        # Method for the Agents to run while learning
        self.step(False)
        return self.get_game_state()

    def close_program(self):
        pygame.quit()
        sys.exit()

    def step(self, render=True, multi=False):
        """
            Perform one step of game emulation.
        """
        print(self.player_1.x)
        # Save endstate as screenshot
        if (self.done):
            if(render): self.init()

        # Check if players are alive and update Map

        if self.game_over(multi):

            print(self.score)
            self.done = True
            

        # Check key inputs
        if render == True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.close_program()
                if event.type == pygame.QUIT:
                    self.close_program()
                self.player_1.handle_input(event)
                        
            self.screen.fill(BG_COLOR)
            self.player_1.draw(self.screen)

        # Update players
        if (self.pause == False):
            if(render):
                self.player_1.do_action(self.get_game_state())
            self.player_1.update()
            self.score += self.step_score()


    def game_over(self, multi):
        # checks, if game is over
        return self.player_1.x == SIZE or self.score < -1000

    def first_init(self):
        self.screenSize = (
            MAP_SIZE[0
                     ] * SCREEN_SCALE, MAP_SIZE[1] * SCREEN_SCALE)

        self.pause = False

    def init(self,render=True):

        self.saveScreen = 0
        self.done = False
        self.score = 0
        self.create_players()
        if render == True:
            self.screen.fill(BG_COLOR)
#------------------------------------------------------------------


class Learn_SinglePlayer (Main):

    def create_players(self):

        self.player_1 = SimplePlayer(MAP_SIZE, COLOR_ONE, SCREEN_SCALE)
        self.multi = False

    def step_score(self):
        
        return -1

class SinglePlayer (Main):

    def create_players(self):

        self.player_1 = SimpleHuman(MAP_SIZE,COLOR_ONE,SCREEN_SCALE,"control_1")
        self.multi = False

    def step_score(self):
        return -1

    def get_end_score(self):
        return self.score

class SimplePlayer(object):

    def __init__(self, mapSize, color,
                 screenScale, control=None):

        # set parameters
        self.screenScale = screenScale
        self.x = 3
        self.y = 0
        self.color = color
        self.alive = True
        self.mapSize = mapSize
        self.init_algorithm()
        self.action = 0

    def lose(self):
        self.close()

    def update(self):
        # Update player position
        if self.x == 0 and self.action == -1:
            pass
        else:
            self.x = self.x + self.action

    def draw(self, screen):
        # render new player position on the screen
        halfScale = self.screenScale / 2
        
        
        for i in range(-halfScale, halfScale):
            for j in range(-halfScale, halfScale):
                pygame.Surface.set_at(
                    screen, (self.x * self.screenScale + i, self.y * self.screenScale + j), self.color)


    def handle_input(self, event):
        pass

    def do_action(self, action, a=None, b=None):
        pass

    def init_algorithm(self):
        pass

class SimpleHuman(SimplePlayer):

    def init_algorithm(self):

            self.actions = {
                "left": pygame.K_a,
                "right": pygame.K_d
            }

    def handle_input(self, event):
        print(self.x)
        if event.type == pygame.KEYDOWN and event.key == self.actions["right"]:
            self.action = 1
        if event.type == pygame.KEYDOWN and event.key == self.actions["left"]:
            self.action = - 1

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
        self.clock.tick(30)
        
