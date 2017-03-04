'''
Created on Feb 26, 2017

@author: marti
'''


from keras.layers import *
from sklearn.externals import joblib
import numpy as np
import copy

class Preprocessor:

    def hubert_loss(self, y_true, y_pred):    # sqrt(1+a^2)-1
        # Its like MSE in intervall (-1,1) and after this linear Error
        err = y_pred - y_true
        return K.mean(K.sqrt(1 + K.square(err)) - 1, axis=-1)

    def dqn_preprocess_state(self, state, state_cnt):
        # converts given state into fitting state for CNN with only matrices
        # creates a diffMap with only zeros except a 1 in the player position
        # doesnt use the rotation

        map = state["map"]
        reward = state["reward"]
        done = state["done"]
        diffMap = np.zeros(shape=(state_cnt[1], state_cnt[2]))
        coords = (int(state["playerPos"][0]), int(state["playerPos"][1]))
        diffMap[coords] = 1
        
        features = np.array([map, diffMap])

        return features, reward, done


    def lfa_constant(self, size):
        self.map_size = size
    def lfa_preprocess_state(self ,state):

        # converts given state into fitting state for the SGD Regressor. Has to be only scalars for the state
        # takes player position as argument and one scalar for each pixel in the environment. The value of theses pixels is max(0, cos(a)/distance
        # where distance is the distance from the pixel to the player and a is
        # the angle between player rotation and vector from position to pixel.


        p = state["playerPos"]
        m = copy.copy(state["map"])
        x = p[0] / self.map_size
        y = p[1] / self.map_size
        # Compute position vector pv

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