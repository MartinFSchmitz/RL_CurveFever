'''
Created on Feb 22, 2017

@author: marti
'''
import matplotlib
import copy
import random
import math
import numpy as np
from SumTree import SumTree
import pygame
from CurveFever import Learn_SinglePlayer
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

""" Q-Learning mit Linearer Funktionsannäherung und den Ideen des DQN"""

STATE_CNT = (2706)  # 52x52 = 2704  + 2 wegen pos
ACTION_CNT = 3  # left, right, straight

MEMORY_CAPACITY = 100  # change to 200 000 (1 000 000 in original paper)

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1

# at this step epsilon will be 0.1  (1 000 000 in original paper)
EXPLORATION_STOP = 500000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

SAVE_XTH_GAME = 5000  # all x games, save the CNN
LEARNING_FRAMES = 10000000

#-------------------- BRAIN ---------------------------


class Brain:

    def __init__(self):

        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.model = []
        a = np.zeros(STATE_CNT)

        for _ in range(ACTION_CNT):
            m = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            m.partial_fit([a], [0])  # any state, just to avoid stupid error
            self.model.append(m)
        self.updateTargetModel()

    def train_single(self, x, y, a, epoch=1, verbose=0):  # old and not used anymore

        #self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

        batch_size = a.size
        for i in range(batch_size):
            action = int(a[i])
            self.model[action].partial_fit(x[i].reshape(1, -1), [y[i][action]])

    def train(self, x, y, a, epoch=1, verbose=0):

        state = [[], [], []]
        target = [[], [], []]
        batch_size = a.size
        for i in range(batch_size):
            action = int(a[i])
            state[action].append(x[i])
            target[action].append(y[i][action])
        for act in range(ACTION_CNT):
            if(state[act] != []):
                self.model[act].partial_fit(state[act], target[act])

    def predict(self, s, target=False):
        # print(s)
        # sometimes scalars instead of  [a b] arrays
        batch_size = np.array(s).size / STATE_CNT
        pred = np.zeros((batch_size, ACTION_CNT))
        # s[0] ist das 0te state-tupel, und pred[0] das 0te tupel von predictions
        # bei m.predict(s)[0]  braucht man die [0] um das Ergebnis, dass ein
        # array ist in ein skalar umzuwandeln

        if target:

            for i in range(batch_size):
                pred[i] = [m.predict(s[i].reshape(1, -1))[0]
                           for m in self.model_]
        else:

            for i in range(batch_size):
                pred[i] = [m.predict(s[i].reshape(1, -1))[0]
                           for m in self.model]

        return pred

    def updateTargetModel(self):
        # self.model_.set_weights(self.model.get_weights())
        a = copy.copy(self.model)
        self.model_ = a


#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01  # epsilon
    a = 0.6  # alpha
    # For PER (Prioritized Experience Replay): (error+e)^a

    def __init__(self, capacity):

        # Sumtree sucht tupel mit O(log n) anstatt O(n) mit Array
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):  # new Sample
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):

        # n = amount of samples in batch
        batch = []
        segment = self.tree.total() / n

        for i in range(n):  # tree is divided into segments of equal size
            a = segment * i  # take one sample out of every segment
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)  # get with O(log n)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):  # Update Priority of a sample
        p = self._getPriority(error)
        self.tree.update(idx, p)

        #-------------------- AGENT ---------------------------


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self):

        self.brain = Brain()
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):  # epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_CNT - 1)
        else:
            return np.argmax(self.brain.predict([s]))

    def observe(self, sample):  # in (s, a, r, s_) format

        error = self._get_sample_Target(sample)
        self.memory.add(error, sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + \
            (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _get_sample_Target(self, sample):
        # sample in (s, a, r, s_) format
        p = agent.brain.predict(sample[0].reshape(1, -1), target=False)[0]
        p_ = agent.brain.predict(sample[3].reshape(1, -1), target=False)[0]
        pTarget_ = agent.brain.predict(
            sample[3].reshape(1, -1), target=True)[0]
        oldVal = p[sample[1]]
        if sample[3] is None:
            target = sample[2]  # target = reward
        else:
            # double DQN (Bellmann Equation)
            target = sample[2] + GAMMA * pTarget_[np.argmax(p_)]
        error = abs(oldVal - target)
        return error

    # Computes (Input, Output, Error) tuples -->Q-Learning happens here
    def _getTargets(self, batch):
        no_state = np.zeros(STATE_CNT)

        states = np.array([o[1][0] for o in batch])  # stores all states
        states_ = np.array([(no_state if o[1][3] is None else o[1][3])
                            for o in batch])  # stores only final states

        p = agent.brain.predict(states, target=False)
        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = np.zeros((len(batch), STATE_CNT))
        y = np.zeros((len(batch), ACTION_CNT))
        z = np.zeros(len(batch))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                # double DQN (Bellmann Equation)
                t[a] = r + GAMMA * pTarget_[i][np.argmax(p_[i])]

            x[i] = s
            y[i] = t
            z[i] = a
            errors[i] = abs(oldVal - t[a])

        return (x, y, z, errors)

    def replay(self):
         # Update Tuples and Errors, than train the SGD Regressor
        batch = self.memory.sample(BATCH_SIZE)
        x, y, a, errors = self._getTargets(batch)
        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y, a)


class RandomAgent:  # Takes Random Action
    # Takes Random Action
    # will be used to fill memory in the beginning
    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self):
        pass

    def act(self, s):
        return random.randint(0, ACTION_CNT - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # error = reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass


#-------------------- ENVIRONMENT ---------------------
class Environment:

    def preprocess_state_old(self):

        # takes only the player position as arguments for the state

        state = game.AI_learn_step()
        p = state["playerPos"]
        x = p[0] / 52.0
        y = p[1] / 52.0
        features = np.array([x, y])
        # if only_state: return features
        return features, state["reward"], state["done"]

    def preprocess_state(self):

        # converts given state into fitting state for the SGD Regressor. Has to be only scalars for the state
        # takes player position as argument and one scalar for each pixel in the environment. The value of theses pixels is max(0, cos(a)/distance
        # where distance is the distance from the pixel to the player and a is
        # the angle between player rotation and vector from position to pixel.

        state = game.AI_learn_step()
        p = state["playerPos"]
        m = copy.copy(state["map"])
        x = p[0] / 52.0
        y = p[1] / 52.0
        # Compute position vector pv
        # cos elementwise ..maybe interesting for more efficiency ?
        pv_x = np.cos(state["playerRot"])
        pv_y = np.sqrt(1 - pv_x)

        for i in xrange(m[0].size):  # y coord
            for j in xrange(m[0].size):  # x coord
                if(m[i][j] == 1):
                    dy = i - y
                    dx = j - x
                    dist = np.sqrt(dx**2 + dy**2)
                    dx = dx / dist
                    dy = dy / dist
                    # scalarproduct
                    cosa = dx * pv_x + dy * pv_y

                    if (dist != 0):
                        res = max(cosa, 0) / dist
                    else:
                        res = 1
                    m[i][j] = res
                # print(m[i][j])

        m = m.reshape(1, -1)[0]

        pos = np.array([x, y])
        features = np.append(pos, m)

        # if only_state: return features
        return features, state["reward"], state["done"]

    def run(self, agent, game, count):

        # run one episode of the game, store the states and replay them every
        # step
        s, r, done = self.preprocess_state()  # 1st frame no action
        R = 0
        k = 4  # step hopper
        counter = 0
        while True:
            # one step of game emulation
            if (counter % k == 0):
                a = agent.act(s)  # agent decides an action
                # converts interval (0,2) to (-1,1)
                game.player_1.action = a - 1
                s_, r, done = self.preprocess_state()
                agent.observe((s, a, r, s_))  # agent adds the new sample
                agent.replay()
                s = s_
            else:
                s, r, done = self.preprocess_state()
            counter += 1
            R += r
            if done:  # terminal state
                break
        print("Total reward:", R)
        return count + (counter / k)
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
        if frame_count >= LEARNING_FRAMES:
            break
        frame_count = env.run(agent, game, frame_count)
        episode_count += 1

        if episode_count % SAVE_XTH_GAME == 0:  # all x games, save the CNN %xtesspiel%savexthgame == 0
            save_counter = episode_count / SAVE_XTH_GAME

            print("Didnt Save model " + str(save_counter) + " to disk")

finally:

    print("Didnt Save FINAL model to disk.")
    print("-----------Finished Process----------")
