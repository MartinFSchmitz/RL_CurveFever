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

""" klassisches Q-Learning mit Linearer Funktionsannaeherung """

# HYPER-PARAMETERS
LOADED_DATA = 'data/lfa_rei/m2_mit_greedy.p'
GAMEMODE = "multi_2" # single, multi_1, multi_2

STATE_CNT = 4
ACTION_CNT = 4  # left, right, straight

NUM_EPISODES = 30000

GAMMA = 0.99
EPSILON = 0.1,
EPSILON_DECAY = 1.0
MAX_EPSILON = 1
MIN_EPSILON = 0.1
ALPHA = 0.05

# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 10000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

#------------------------------------------------------------------

game = RL_Algo.init_game()
pre = LFAPreprocessor(STATE_CNT)
game.init(render=False)
state, _,_ = pre.lfa_preprocess_state_feat(game.AI_learn_step())

# Hack to not always set STATE_CNT manually
STATE_CNT = len(pre.lfa_preprocess_state_feat( game.get_game_state())[0])
#------------------------------------------------------------------


class Brain():
    """
    Value Function approximator.
    """

    def __init__(self, init_state):

        self.models = []
        for _ in range(ACTION_CNT):

            model = np.zeros(STATE_CNT)

            self.models.append(model)

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
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
        Updates the Brain parameters for a given state and action towards
        the target y.
        """
        # print(self.models)
        self.models[a] = self.models[a] + ALPHA * y * s
        # print(self.models[a])


#------------------------------------------------------------------

def make_epsilon_greedy_policy(Brain, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        Brain: An Brain that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = Brain.predict(observation)
        best_action = np.argmax(q_values)
        val = q_values[best_action]
        # print(val)
        A[best_action] += (1.0 - epsilon)
        return A, val
    return policy_fn

#------------------------------------------------------------------


def q_learning(game, Brain):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        Brain: Action-Value function Brain
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = None

    for i_episode in range(NUM_EPISODES):

        # The policy we're following
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
            math.exp(-LAMBDA * i_episode)
        policy = make_epsilon_greedy_policy(
            Brain, epsilon, ACTION_CNT)

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
            q_values_next = Brain.predict(next_state)
            # print(q_values_next)

            # Use this code for Q-Learning
            # Q-Value TD Target
            if not done:
                td_target = reward + GAMMA * np.max(q_values_next)

            else:
                td_target = reward

            td_error = (td_target - old_val)
            # print(q_values_next)
            # Update the function approximator using our target

            Brain.update(state, action, td_error)
            if done:
                print("done episode: ", i_episode, "time:", t)
                rewards.append(t)
                if i_episode % 5000 == 0:
                    if (GAMEMODE == "multi_2"):
                        pickle.dump(agent.policy_brain.model, open(
                        'data/lfa_rei/training_pool/agent_' + str(save_counter) +'.p', 'wb'))   
                    pickle.dump(Brain.models, open(
                        'data/lfa/save.p', 'wb'))
                    RL_Algo.make_plot(rewards, 'lfa', 100, save_array=True)

                break

            state = next_state
    pickle.dump(Brain.models, open(
                        'data/lfa/save.p', 'wb'))
    RL_Algo.make_plot(rewards, 'lfa', 100, save_array=True)
    return stats
#------------------------------------------------------------------


game = RL_Algo.init_game()
Brain = Brain(state)
stats = q_learning(game, Brain)
