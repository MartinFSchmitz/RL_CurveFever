'''
Created on Mar 8, 2017

@author: marti
'''

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import random
from CurveFever import Learn_SinglePlayer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#------------------------------------------------------------------
def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true           #Its like MSE in intervall (-1,1) and after this linear Error
    #self.test = False
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

class Brain:

    def _createModel(self,input, output, act_fun): # Creating a CNN
        self.state_Cnt = input
        
        model = Sequential()
        # creates layer with 32 kernels with 8x8 kernel size, subsample = pooling layer
        #relu = rectified linear unit: f(x) = max(0,x), input will be 2 x Mapsize
    
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(input),dim_ordering='th'))    
        #model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu',input_shape=(input),dim_ordering='th'))
        model.add(Flatten())
        model.add(Dense(output_dim=256, activation='relu'))
    
        model.add(Dense(output_dim=output, activation=act_fun))
    
        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        model.compile(loss='mse', optimizer=opt)
        return model
    
    def predictOne(self, s, target = False):
        state =s.reshape(1, self.state_Cnt[0], self.state_Cnt[1], self.state_Cnt[2])
        return self.predict(state, target).flatten()
    
#------------------------------------------------------------------
def get_random_equal_state(sample):

    # Gets one random of the 8 equivalent mirrored and rotated states, to
    # the current state
    s, a, r, s_ = sample
    rnd = random.randint(0, 7)

    if (rnd % 2 == 1):
        s = np.fliplr(s)
        s_ = np.fliplr(s_)
        a = -1*(a-1)+1 # mirror a, too
    rotate = rnd / 2
    if (rotate < 0):
        #for i in xrange(STATE_CNT[0]):  # Maps einzeln rotieren
            #s[i] = np.rot90(s[i], rotate)
            #s_[i] = np.rot90(s[i], rotate)
            
        s[0] = np.rot90(s[0], rotate)
        s_[0] = np.rot90(s[0], rotate)
        s[1] = np.rot90(s[1], rotate)
        s_[1] = np.rot90(s[1], rotate)
        
    return (s,a,r,s_)
#------------------------------------------------------------------  
def init_game():
        # init Game Environment
    game = Learn_SinglePlayer()
    game.first_init()
    game.init( render = False)
    return game

def save_model(model, file, name):
    # serialize weights to HDF5
    model.save_weights(
    "data/" + file + "/model_" + name + ".h5")
    print("Saved model " + name + " to disk")
    if name == 'final':
        # serialize model to JSON
        model_json = model.to_json()
        with open("data/" + file + "/model.json", "w") as json_file:
            json_file.write(model_json)

def make_plot(x, name, step):
    
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
     
    plt.plot( episodes,reward_array,linewidth=0.2,color='g')
    plt.plot(episodes_step,reward_step_array,linewidth=1.5,color = 'r')

    plt.xlabel('Number of episode')
    plt.ylabel('Reward')
    plt.title(name.upper() + ': Rewards per episode')
    plt.grid(True)
    plt.savefig("data/" + name + "/" + name +"_plot.png")
    print("made plot...")


