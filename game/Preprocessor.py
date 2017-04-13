'''
Created on Feb 26, 2017

@author: marti
'''


from keras.layers import *
import numpy as np
import copy

class CNNPreprocessor:

    def hubert_loss(self, y_true, y_pred):    # sqrt(1+a^2)-1
        # Its like MSE in intervall (-1,1) and after this linear Error
        err = y_pred - y_true
        return K.mean(K.sqrt(1 + K.square(err)) - 1, axis=-1)
    
    def __init__(self,state_cnt):
        #self.zero_map = np.zeros(shape=(state_cnt[1], state_cnt[2]))
        #self.spur_map = np.zeros(shape=(state_cnt[1], state_cnt[2]))
        self.state_cnt = state_cnt
    
    def cnn_preprocess_state(self, state, multi_player = False):
        # converts given state into fitting state for CNN with only matrices
        # creates a diffMap with only zeros except a 1 in the player position
        # doesnt use the rotation

        map = state["map"]
        reward = state["reward"]
        done = state["done"]
        player_map = np.zeros(shape=(self.state_cnt[1], self.state_cnt[2]))
        player_coords = (int(state["playerPos"][0]), int(state["playerPos"][1]))
        player_map[player_coords] = 1
        #self.spur_map[player_coords] = 1
        if multi_player:
            print(multi_player)
            opponent_map = self.zero_map
            opponent_coords = (int(state["opponentPos"][0]), int(state["opponentPos"][1]))
            opponent_map[opponent_coords] = 1
            features = np.array([map, player_map,opponent_map])
        else: features = np.array([map, player_map])

        return features, reward, done
    
    def cnn_preprocess_state_2(self, state, multi_player = False):
        # converts given state into fitting state for CNN with only matrices
        # creates a diffMap with only zeros except a 1 in the player position
        # doesnt use the rotation

        player_map = copy.copy(state["map"])
        reward = state["reward"]
        done = state["done"]
        player_coords = (int(state["playerPos"][0]), int(state["playerPos"][1]))
        player_map[player_coords] = 10
        features = np.array([player_map])
        
        return features, reward, done
    
class LFAPreprocessor:    
    
    def __init__(self, size):
        self.map_size = size

    def lfa_preprocess_state(self ,state, multi_player = False):

        # converts given state into fitting state for the SGD Regressor. Has to be only scalars for the state
        # takes player position as argument and one scalar for each pixel in the environment. The value of theses pixels is max(0, cos(a)/distance
        # where distance is the distance from the pixel to the player and a is
        # the angle between player rotation and vector from position to pixel.
        features = self.get_features(state["playerPos"],state["playerRot"],state["map"])
        if (multi_player):
            opponent_features = self.get_features(state["opponentPos"],state["opponentRot"],state["map"])
            features = np.append(features, opponent_features)
        return features, state["reward"], state["done"]

    def get_features(self, pos, rot, map):

        m = copy.copy(map)
        x = pos[0] / self.map_size
        y = pos[1] / self.map_size 
        # Compute position vector pv

        pv_x = np.cos(rot)
        pv_y = np.sqrt(1 - pv_x)
        for i in range(m[0].size):  # y coord
            for j in range(m[0].size):  # x coord
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
        x = 0.5 - x
        y = 0.5 - y
        abs =  np.sqrt((x**2)+(y**2))
        pos = np.array([abs,x, y])
        features = np.append(pos, m)

        return features
    
    def lfa_preprocess_state_2(self ,state, multi_player = False):

        p = state["playerPos"]
        dx = 10 - p[0]
        dy = 10 - p[1]
        abs =  np.sqrt((dx**2)+(dy**2))
        features = np.array([abs, dy, dx])
        return features, state["reward"], state["done"]

    
    