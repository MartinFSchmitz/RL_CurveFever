'''
Created on 17.02.2017

@author: Martin
'''

#--- enable this to run on GPU
#import os    
#os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  
#---

import random, numpy, math
from SumTree import SumTree
import pygame
from CurveFever import Learn_SinglePlayer
from keras.models import load_model
from keras.utils.np_utils import binary_logloss
from keras import optimizers



def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true           #Its like MSE in intervall (-1,1) and after this linear Error
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

STATE_CNT  = (2,52,52) # 2=Map + diffMap, height, width
ACTION_CNT = 3 # left, right, straight

MEMORY_CAPACITY = 100000 # change to 200 000 (1 000 000 in original paper)

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1

EXPLORATION_STOP = 500000   # at this step epsilon will be 0.1  (1 000 000 in original paper)
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

SAVE_XTH_GAME = 5000  # all x games, save the CNN
LEARNING_FRAMES = 10000000

#-------------------- BRAIN ---------------------------

class Brain:
    def __init__(self):

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self): # Creating a CNN
        model = Sequential()

        # creates layer with 32 kernels with 8x8 kernel size, subsample = pooling layer
        #relu = rectified linear unit: f(x) = max(0,x), input will be 2 x Mapsize
        # I divided every parameter to 2, and made the kernel sizes 30%smaller because the input size is 30% smaller as well
        
        #model.add(Convolution2D(64, 8, 8, subsample=(4,4), activation='relu', input_shape=(STATE_CNT)))
        model.add(Convolution2D(16, 6, 6, subsample=(4,4), activation='relu', input_shape=(STATE_CNT)))     
        model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(32, 2, 2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=256, activation='relu'))

        model.add(Dense(output_dim=ACTION_CNT, activation='linear'))

        opt = RMSprop(lr=0.00025) #RMSprob is a popular adaptive learning rate method 
        model.compile(loss=hubert_loss, optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        # x=input, y=target, batch_size = Number of samples per gradient update
        #nb_epoch = number of the epoch, 
        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        #return self.predict(s.reshape(1, (2, 80+2, 80+2)), target).flatten() #8 0 = mapsize
        return self.predict(s.reshape(1, 2, 50+2, 50+2), target).flatten() #8 0 = mapsize   try like this...
    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01 #epsilon
    a = 0.6 #alpha
    # For PER (Prioritized Experience Replay): (error+e)^a

    def __init__(self, capacity):
        
        #Sumtree sucht tupel mit O(log n) anstatt O(n) mit Array
        self.tree = SumTree(capacity) 

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample): #new Sample
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        
        # n = amount of samples in batch
        batch = []
        segment = self.tree.total() / n

        for i in range(n): # tree is divided into segments of equal size
            a = segment * i # take one sample out of every segment
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s) #get with O(log n)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error): # Update Priority of a sample
        p = self._getPriority(error)
        self.tree.update(idx, p)
        
        #-------------------- AGENT ---------------------------


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self):

        self.brain = Brain()
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s): # epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_CNT-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch): # Computes (Input, Output, Error) tuples -->Q-Learning happens here
        no_state = numpy.zeros(STATE_CNT)

        states = numpy.array([ o[1][0] for o in batch ]) # stores all states
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ]) # stores only final states
        
        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)
        
        x = numpy.zeros((len(batch), STATE_CNT[0], STATE_CNT[1], STATE_CNT[2]))
        y = numpy.zeros((len(batch), ACTION_CNT))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN (Bellmann Equation)

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self): # Update Tuples and Errors, than train the CNN   
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent: # Takes Random Action

    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self):
        pass

    def act(self, s):
        return random.randint(0, ACTION_CNT-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # error = reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass
    

#-------------------- ENVIRONMENT ---------------------
class Environment:

    def preprocess_state(self):
        
        state = game.AI_learn_step()
        map = state["map"]
        reward = state["reward"]
        done = state["done"]
        diffMap = numpy.zeros(shape=(STATE_CNT[1],STATE_CNT[2]))
        coords = (int(state["playerPos"][0]),int(state["playerPos"][1]))
        diffMap[coords] = 1
        
        return map ,diffMap, reward, done

    def run(self, agent, game, count):                

        map ,diffMap , r, done = self.preprocess_state() # 1st frame no action
        s = numpy.array([map, diffMap])       
        R = 0
        k = 4 # step hopper
        counter = 0
        while True:         
            
            if (counter % k == 0):
                a = agent.act(s) # agent decides an action        
                game.player_1.action = a-1 # converts interval (0,2) to (-1,1)
                map ,diffMap, r, done = self.preprocess_state()
                s_ = numpy.array([map, diffMap])#last two screens
                agent.observe( (s, a, r, s_) ) # agent adds the new sample
                agent.replay()                
                s = s_
            else: 
                map ,diffMap, r, done = self.preprocess_state()   
            counter+=1
            R+=r
            if done:    #terminal state  
                break
        #print("Total reward:", R)
        return count + (counter/k)
#-------------------- MAIN ----------------------------

env = Environment()

# init Game Environment
game = Learn_SinglePlayer()   
game.first_init()
game.init(game, False)


agent = Agent()
randomAgent = RandomAgent()

try:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        env.run(randomAgent, game, 0)
        print(randomAgent.exp, "/", MEMORY_CAPACITY)

    agent.memory = randomAgent.memory

    randomAgent = None

    print("Starting learning")
    count = 0
    save_counter = 0
    while True:
        if count >= LEARNING_FRAMES: break
        count = env.run(agent, game, count)
                    # serialize model to JSON
        #model_json = agent.brain.model.to_json()
        #with open("model.json", "w") as json_file:
            #json_file.write(model_json)
        # serialize weights to HDF5

        if save_counter % SAVE_XTH_GAME == 0: # all x games, save the CNN
            save_counter += 1
            agent.brain.model.save_weights("data/dqn/model_" + str(save_counter) + ".h5")
            print("Saved model " + str(save_counter) + " to disk")

finally:
            # serialize model to JSON
        model_json = agent.brain.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        #serialize weights to HDF5
        agent.brain.model.save_weights("data/dqn/model_end.h5")
        print("Saved FINAL model to disk.")
        print("-----------Finished Process----------")
