'''
Created on Feb 22, 2017

@author: marti
'''


import random,  math
import numpy as np
from SumTree import SumTree
import pygame
from CurveFever import Learn_SinglePlayer
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

STATE_CNT  = (2) # 2=Map + diffMap, height, width
ACTION_CNT = 3 # left, right, straight

MEMORY_CAPACITY = 1000 # change to 200 000 (1 000 000 in original paper)

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

        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        models = []
        for _ in range(ACTION_CNT):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit(np.array([0.5,0.5]), [0]) #any state, just to avoid stupid error
            models.append(model)
            
        return models

    def train(self, x, y, a, epoch=1, verbose=0):

        #self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)
        self.models[a].partial_fit([x], [y])
        
    def predict(self, s, target=False):
        if target:
            
            return np.array([m.predict(s)[0] for m in self.model_])
        else:
            print(s)
            #return self.model.predict(s)
            return np.array([m.predict(s)[0] for m in self.model])


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
            return np.argmax(self.brain.predict(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        
        x, y, z, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch): # Computes (Input, Output, Error) tuples -->Q-Learning happens here
        no_state = np.zeros(STATE_CNT)

        states = np.array([ o[1][0] for o in batch ]) # stores all states
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ]) # stores only final states
        
        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)
        
        x = np.zeros((len(batch), STATE_CNT))
        y = np.zeros((len(batch), ACTION_CNT))
        z = np.zeros(len(batch))
        errors = np.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ np.argmax(p_[i]) ]  # double DQN (Bellmann Equation)

            x[i] = s
            y[i] = t
            z[i] = a
            errors[i] = abs(oldVal - t[a])

        return (x, y, z, errors)

    def replay(self): # Update Tuples and Errors, than train the CNN   
        batch = self.memory.sample(BATCH_SIZE)
        x, y, a, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y, a)

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
        p = state["playerPos"]
        x = p[0]/52.0
        y = p[1]/52.0
        features = np.array([x,y])
        #if only_state: return features
        return features, state["reward"],state["done"]

    def run(self, agent, game, count):                

        s, r, done = self.preprocess_state() # 1st frame no action   
        R = 0
        k = 4 # step hopper
        counter = 0
        while True:         
            
            if (counter % k == 0):
                a = agent.act(s) # agent decides an action        
                game.player_1.action = a-1 # converts interval (0,2) to (-1,1)
                s_, r, done = self.preprocess_state()
                agent.observe( (s, a, r, s_) ) # agent adds the new sample
                agent.replay()                
                s = s_
            else: 
                s, r, done = self.preprocess_state()       
            counter+=1
            R+=r
            if done:    #terminal state  
                break
        print("Total reward:", R)
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
    frame_count = 0
    episode_count = 0
    
    while True:
        if frame_count >= LEARNING_FRAMES: break
        frame_count = env.run(agent, game, frame_count)
        episode_count += 1

        if episode_count % SAVE_XTH_GAME  == 0: # all x games, save the CNN %xtesspiel%savexthgame == 0
            save_counter =  episode_count / SAVE_XTH_GAME

            print("Didnt Save model " + str(save_counter) + " to disk")

finally:

        print("Didnt Save FINAL model to disk.")
        print("-----------Finished Process----------")

