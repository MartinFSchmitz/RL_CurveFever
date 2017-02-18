'''
Created on 17.02.2017

@author: Martin
'''
import cPickle as pickle
import itertools
#import matplotlib
import numpy as np
import sys
import math
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import pygame
from game.CurveFever import Learn_SinglePlayer


# HYPER-PARAMETERS
STATE_CNT  = (2, 52, 52) # 2=Map + diffMap, height, width
ACTION_CNT = 3 # left, right, straight

NUM_EPISODES = 1000 
GAMMA = 0.99
EPSILON = 0.1, 
EPSILON_DECAY = 1.0
MAX_EPSILON = 1
MIN_EPSILON = 0.1

EXPLORATION_STOP = 500000   # at this step epsilon will be 0.1  (1 000 000 in original paper)
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

#------------------------------------------------------------------

    
def preprocess_state(onlyState = True):
    
    state = game.AI_learn_step()
    p = state["playerPos"]
    x = p[0]
    y = p[1]
    r = state["playerRot"]
    features = np.array([x,y,r])
    if onlyState: return features
    else: return features, state["reward"],state["done"]


# init Game Environment
game = Learn_SinglePlayer()   
game.first_init()
game.init(game, render = False)
state = preprocess_state()
#------------------------------------------------------------------

# weird Preprocessing...


# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
#observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
observation_examples = []
for i in range (0,1000):
    game.init(game, render = False)
    s = preprocess_state()
    observation_examples.append(s)
observation_examples = np.array (observation_examples)
#print(observation_examples)

#observation_examples.reshape(1, -1)
scaler = sklearn.preprocessing.StandardScaler()

scaler.fit(observation_examples)
# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))
pickle.dump(featurizer, open('lfa/featurizer.p', 'wb'))



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
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(init_state)], [0]) #any state, just to avoid stupid error
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]
    
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
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])
        
        
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
        A[best_action] += (1.0 - epsilon)
        return A
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
    stats=None
    #stats = plotting.EpisodeStats(
    #    episode_lengths=np.zeros(num_episodes),
    #    episode_rewards=np.zeros(num_episodes))    
    
    for i_episode in range(NUM_EPISODES):
        
        
        # The policy we're following
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA *i_episode)
        policy = make_epsilon_greedy_policy(
            estimator, epsilon , ACTION_CNT)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        #last_reward = stats.episode_rewards[i_episode - 1]
        #sys.stdout.flush()
        
        # Reset the environment and pick the first action
        game.init(game, False)
        state = preprocess_state()
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            game.player_1.action = action-1 # converts interval (0,2) to (-1,1)
            # Take a step
            next_state, reward, done = preprocess_state( onlyState = False)
            
            # Update statistics
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t
            
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + GAMMA * np.max(q_values_next)
            
            # Update the function approximator using our target
            estimator.update(state, action, td_target)
                
            if done:
                print("done episode: ", i_episode, "time:", t )
                if i_episode % 100 == 0: pickle.dump(estimator.models, open('lfa/save.p', 'wb'))
                break
                
            state = next_state
    
    return stats
#------------------------------------------------------------------


estimator = Estimator(state)
stats = q_learning(game, estimator)
