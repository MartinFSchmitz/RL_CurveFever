'''
Created on 17.02.2017

@author: Martin
'''
import pickle
import itertools
import matplotlib
import numpy as np
import sys
import math

import pygame
from CurveFever import Learn_SinglePlayer
import RL_Algo
from Preprocessor import LFAPreprocessor

""" Q-Learning with linear function approximation """

# used https://github.com/dennybritz/ as reference

# HYPER-PARAMETERS

# Load already trained models to continue training:
LOADED_DATA = None #'data/lfa/save.p' # note: change Epsilon, when you load data
# Train for singleplayer or multiplayer
GAMEMODE = "single"
#print episode results
PRINT_RESULTS = False
ALGORITHM = "lfa"

STATE_CNT = 0 # will be changed later
# amount of possible actions for the agent
ACTION_CNT = 4

SIZE = 30
NUM_EPISODES = 50000

SAVE_XTH_GAME = 3000

GAMMA = 0.99
# parameters for decreasing epsilon
EPSILON = 0.1,
EPSILON_DECAY = 1.0
MAX_EPSILON = 1
MIN_EPSILON = 0.1
# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 25000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

# learning parameter
ALPHA = 0.008


#------------------------------------------------------------------
# initialize game environment and preprocessor
game = RL_Algo.init_game(GAMEMODE, ALGORITHM)
pre = LFAPreprocessor(SIZE)
game.init(render=False)
# Hack to not always set STATE_CNT manually
STATE_CNT = len(pre.lfa_preprocess_state_feat( game.get_game_state())[0])
print("STATE_CNT = ",STATE_CNT)
state, _,_ = pre.lfa_preprocess_state_feat(game.AI_learn_step())

#------------------------------------------------------------------


class Estimator():
    """
    Value Function approximator. Class to store and modify linear Q-Function model
    """

    def __init__(self, init_state):
        """ initialize LFA model """
        # Creating one model for every action in action space
        if (LOADED_DATA == None):
            self.models = []
            for _ in range(ACTION_CNT):
                model = np.zeros(STATE_CNT)
                self.models.append(model)
        else:
            # use previously trained model when given
            with open(LOADED_DATA, 'rb') as pickle_file:
                self.models = pickle.load(pickle_file)
            print(self.models)

    def predict(self, s, a=None):
        
        """
        Makes value function predictions.

        Input:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Output
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        if not a:
            return np.array([np.inner(m, s) for m in self.models])
        else:
            return np.inner(self.models[a], s)

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        self.models[a] = self.models[a] + ALPHA * y * s

#------------------------------------------------------------------

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        val = q_values[best_action]
        # print(val)
        A[best_action] += (1.0 - epsilon)
        return A, val
    return policy_fn

#------------------------------------------------------------------


def q_learning(game, estimator):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    rewards = []
    for episode_count in range(NUM_EPISODES):

        # The policy we're following
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
            math.exp(-LAMBDA * episode_count)
        policy = make_epsilon_greedy_policy(
            estimator, epsilon, ACTION_CNT)

        # Reset the environment and pick the first action
        game.init(render=False)
        state,_,_ = pre.lfa_preprocess_state_feat(game.AI_learn_step())
        # One step in the environment
        for t in itertools.count():

            # Choose an action to take
            action_probs, old_val = policy(state)
            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)
            # converts interval (0,2) to (-1,1)
            game.player_1.action = action
            # Take a step
            next_state, reward, done = pre.lfa_preprocess_state_feat(game.AI_learn_step())

            # TD Update
            q_values_next = estimator.predict(next_state)
            # print(q_values_next)

            # Use this code for Q-Learning
            # Q-Value TD Target
            if not done:
                td_target = reward + GAMMA * np.max(q_values_next)

            else:
                td_target = reward

            td_error = (td_target - old_val)
            # Update the function approximator using our target

            estimator.update(state, action, td_error)
            # save models and statistics after certain amount of episodes
            if done:
                print("done episode: ", episode_count, "time:", t)
                rewards.append(t)
                
                if episode_count % SAVE_XTH_GAME == 0:
                    save_counter = int(episode_count / SAVE_XTH_GAME)
                    RL_Algo.make_plot(rewards, 'lfa', 100, save_array=True)
                    
                    if (GAMEMODE == "multi_2"):
                        file = 'data/lfa/training_pool/agent_' + str(save_counter) +'.p'  
                    else:
                        file = 'data/lfa/save.p'
                    pickle.dump(estimator.models, open(
                        file, 'wb'))
                break

            state = next_state

#------------------------------------------------------------------

# initialize game and value function
game = RL_Algo.init_game(GAMEMODE,ALGORITHM)
estimator = Estimator(state)
q_learning(game, estimator)
