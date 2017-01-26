'''
Created on 31.12.2016

@author: Martin
'''
from _elementtree import tostring

""" Trains an agent with REINFORCE Policy Gradients on Zatacka."""
import pygame
import game.GameMode
import numpy as np
import cPickle as pickle

# init Game Environment
game = game.GameMode.SinglePlayer()   
pygame.init()    
game.firstInit()
game.screen = pygame.display.set_mode(game.screenSize)
game.clock = pygame.time.Clock()
game.init(game) 


# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 100 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True

# model initialization
D = (game.mapSize[0]+2) * (game.mapSize[1]+2) # input dimensionality: 80x80 grid (as long as Mapsize = 80x80)
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {} # W1 = first Layer weights , W2 = second Layer weights
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

episode_number = 0

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

# book-keeping
def book_keeping():        
    if episode_number % 100 == 0: pickle.dump(model, open('KNNs/save_' + str(episode_number) + '.p', 'wb'))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x): # Matrix-Mult: LxM * MxN = LxN
    h = np.dot(model['W1'], x) # Kantengewichte * Input = Hiddenlayer-Output as H x 1 Vector
    # ([ H x (D*D)] * [(D*D) x 1] = [H x 1] )
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h) # [1 x H] * [H x 1] = 1 x 1 -->Output
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2']) #outer a,b = dot a^T,b (glaub ich)
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)      # epx = xs (also observation-list) nach vstack
    return {'W1':dW1, 'W2':dW2}


xs = [] # List to append: x = observation (pixel) 
hs = [] # List to append: h = Hidden Layer Output, each step
dlogps = [] # List to append: dlogp = grad that encourages the action, each step
drs = [] # List to append: reward = currend reward , each step

running_reward = None
reward_sum = 0
episode_number = 0
action = 1

while(True):
    
    # step the environment and get new measurements
    game.players[0].action = action
    state, reward, done = game.AiStep()
    x  = state["map"].flatten()
    
    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    #action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    
    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    #dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    

    reward_sum += reward    
    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    
    
    #-------------------------------------
    
    
    
    if done == True: book_keeping()
    """ Just For Test """
    if render:
        pygame.display.update()
        game.clock.tick(30)


