'''
Created on Mar 8, 2017

@author: marti
'''

import numpy as np
import random
from CurveFever import Learn_SinglePlayer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

#------------------------------------------------------------------
def get_random_equal_state(sample):

    """ Gets one random state of the 8 equivalent mirrored and rotated states, to
    the current state """
    s, a, r, s_ = sample
    rnd = random.randint(0, 7)
    """ note: only for curve so far """
    if (rnd % 2 == 1):
        s = np.fliplr(s)
        s_ = np.fliplr(s_)
        a = -1*(a-1)+1 # mirror a, too
    rotate = rnd / 2
    if (rotate < 0):
        # Maps einzeln rotieren
        #for i in range(STATE_CNT[0]):  
            #s[i] = np.rot90(s[i], rotate)
            #s_[i] = np.rot90(s[i], rotate)
            
        s[0] = np.rot90(s[0], rotate)
        s_[0] = np.rot90(s[0], rotate)
        s[1] = np.rot90(s[1], rotate)
        s_[1] = np.rot90(s[1], rotate)
        
    return (s,a,r,s_)
#------------------------------------------------------------------  
def init_game():
    """ init Game Environment """
    game = Learn_SinglePlayer()
    game.first_init()
    game.init( render = False)
    return game

def save_model(model, file, name):
    """saves model structure to json and weights to h5 file """
    model.save_weights(
    "data/" + file + "/model_" + name + ".h5")
    print("Saved model " + name + " to disk")
    if name == 'final':
        # serialize model to JSON
        model_json = model.to_json()
        with open("data/" + file + "/model.json", "w") as json_file:
            json_file.write(model_json)

def make_plot(x, name, step, save_array = False):
    """ creates and saves plot of given arrays of rewards
    Input:
    x: array with values
    name: name of directory and file name
    step: steps of the regression curve 
    save_array: bool if the array should be saved
    Output:
    Saved Plot
    (if save_array): Saved array of values """
    
    if ( save_array == True):
        pickle.dump(np.asarray(x), open(
        'data/'+ name +'/'+'reward_array'+'.p', 'wb'))
    step_x = []
    rewards = 0
    for i in range (len(x)): #xrange
        rewards += x[i]
        if i % step == 0 and i != 0:
            step_x.append(rewards/step)
            rewards = 0
    reward_array = np.asarray(x)
    episodes = np.arange(0, reward_array.size, 1)
    reward_step_array = np.asarray(step_x)   
    episodes_step = np.arange(step/2, reward_array.size-step/2 , step)    
     
    plt.plot( episodes,reward_array,linewidth=0.1,color='g')
    plt.plot(episodes_step,reward_step_array,linewidth=1.5,color = 'r')

    plt.xlabel('Number of episode')
    plt.ylabel('Reward')
    plt.title(name.upper() + ': Rewards per episode')
    plt.grid(True)
    plt.savefig("data/" + name + "/" + name +"_plot.png")
    print("made plot...")


