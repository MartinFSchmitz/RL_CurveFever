'''
Created on May 24, 2017

@author: marti
'''
import sys
import pygame
import Map
from TronPlayer import *
from CurveFever import Learn_MultyPlayer_step_2
import RL_Algo
from Preprocessor import LFAPreprocessor

# Constants
COLOR_ONE = (255, 0, 0)
COLOR_TWO = (0, 255, 0)
COLOR_THREE = (0, 0, 255)
BG_COLOR = (0, 0, 0)
SIZE = 20 # Size of the Game Board  #min 34 with 3 conv Layers
MAP_SIZE = (SIZE, SIZE)
SCREEN_SCALE = 10
LEARNING_EPISODES = 10000
SAVE_XTH_GAME = 500

class Environment:
    
    def __init__(self,p1,p2):       
        """ init Game Environment """
        self.game = Learn_MultyPlayer_step_2()
        self.game.first_init()
        self.pre = LFAPreprocessor(SIZE)
        
        self.game.set_players(p1,p2)
        self.game.init( render = False)

    def run(self):
        self.game.init(render=False)
        state, reward, done = self.pre.lfa_preprocess_state_feat(
            self.game.AI_learn_step())
        R = 0
        while True:
            player_1.do_action(state)
            
            next_state, reward, done = self.pre.lfa_preprocess_state_feat(
                self.game.AI_learn_step())
            state = next_state
            R += reward
            if self.done: break
            
        return R

player_1 = LFA_REI_Player_Tron(MAP_SIZE,COLOR_ONE,SCREEN_SCALE)
player_2 = GreedyPlayer_Tron(MAP_SIZE, COLOR_THREE, SCREEN_SCALE)
env = Environment(player_1,player_2)



rewards = []
try:
    print("Starting learning")

    episode_count = 0

    while True:
        if episode_count >= LEARNING_EPISODES:
            break
        episode_reward = env.run()
        rewards.append(episode_reward)

        episode_count += 1

        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save the CNN

            save_counter = episode_count / SAVE_XTH_GAME
            RL_Algo.make_plot(rewards, 'lfa_rei', 100)
            #pickle.dump(agent.policy_brain.model, open(
                        #'data/lfa_rei/save.p', 'wb'))

finally:
    # make plot
    RL_Algo.make_plot(rewards, 'lfa_rei', 100, save_array=True)
    #pickle.dump(agent.policy_brain.model, open(
        #'data/lfa_rei/save.p', 'wb'))
    print("-----------Finished Process----------")
