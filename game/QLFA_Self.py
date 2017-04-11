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

""" klassisches Q-Learning mit Linearer Funktionsannaeherung """

# HYPER-PARAMETERS
STATE_CNT = 3
ACTION_CNT = 3  # left, right, straight

NUM_EPISODES = 1000000

GAMMA = 0.99
EPSILON = 0.1,
EPSILON_DECAY = 1.0
MAX_EPSILON = 1
MIN_EPSILON = 0.1
ALPHA = 0.0001

# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 10000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

#------------------------------------------------------------------


def preprocess_state(only_state=True):

    state = game.AI_learn_step()
    p = state["playerPos"]
    dx = 10 - p[0]
    dy = 10 - p[1]
    abss =  math.sqrt((dx**2)+(dy**2))
    features = np.array([abss, dy, dx])
    if only_state:
        return features
    else:
        return features, state["reward"], state["done"]


# init Game Environment
game = Learn_SinglePlayer()
game.first_init()
game.init(render=False)
state = preprocess_state()

#------------------------------------------------------------------


class Estimator():
    """
    Value Function approximator. 
    """

    def __init__(self, init_state):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(ACTION_CNT):

            model = np.zeros(STATE_CNT) + 0.5

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
            return np.array([ np.inner(m,s) for m in self.models])
        else:
            np.inner(self.models[a],s)


    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """ 
        #print(self.models)
        self.models[a] = self.models[a] + ALPHA * y * s
        #print(self.models[a]) 


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
        #print(val)
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

    # Keeps track of useful statistics
    stats = None
    # stats = plotting.EpisodeStats(
    #    episode_lengths=np.zeros(num_episodes),
    #    episode_rewards=np.zeros(num_episodes))
    rewards = []
    for i_episode in range(NUM_EPISODES):

        # The policy we're following
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
            math.exp(-LAMBDA * i_episode)
        policy = make_epsilon_greedy_policy(
            estimator, epsilon, ACTION_CNT)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        #last_reward = stats.episode_rewards[i_episode - 1]
        # sys.stdout.flush()

        # Reset the environment and pick the first action
        game.init(render = False)
        state = preprocess_state()
        # One step in the environment
        for t in itertools.count():

            # Choose an action to take
            action_probs, old_val = policy(state)
            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)
            # converts interval (0,2) to (-1,1)
            game.player_1.action = action
            # Take a step
            next_state, reward, done = preprocess_state(only_state=False)

            # Update statistics
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t

            # TD Update
            q_values_next = estimator.predict(next_state)
            #print(q_values_next)

            # Use this code for Q-Learning
            # Q-Value TD Target
            if not done:  
                td_target = reward + GAMMA * np.max(q_values_next)
            else:
                td_target = reward

            td_error = ( td_target - old_val)
            # print(q_values_next)
            # Update the function approximator using our target
            estimator.update(state, action, td_error)
            if done:
                print("done episode: ", i_episode, "time:", t)
                rewards.append(t)
                if i_episode % 10000 == 0:
                    pickle.dump(estimator.models, open(
                        'data/lfa/save.p', 'wb'))
                    RL_Algo.make_plot( rewards, 'lfa', 1000)  
  
                break

            state = next_state

    return stats
#------------------------------------------------------------------

game = RL_Algo.init_game()
estimator = Estimator(state)
stats = q_learning(game, estimator)