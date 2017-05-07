'''
Created on Feb 26, 2017

@author: marti
'''


from keras.layers import *
import numpy as np
import copy
import math
""" Module with several different Preprocessors """

""" Preprocessors for CNNs """

class CNNPreprocessor:

    def hubert_loss(self, y_true, y_pred):    # sqrt(1+a^2)-1
        # Its like MSE in intervall (-1,1) and after this linear Error
        err = y_pred - y_true
        return K.mean(K.sqrt(1 + K.square(err)) - 1, axis=-1)

    def __init__(self, state_cnt):
        #self.zero_map = np.zeros(shape=(state_cnt[1], state_cnt[2]))
        self.spur_map = np.zeros(shape=(state_cnt[1], state_cnt[2]))
        self.state_cnt = state_cnt
        self.begin = True

    def cnn_preprocess_state(self, state, multi_player=False):
        # converts given state into fitting state for CNN with only matrices
        # creates a diffMap with only zeros except a 1 in the player position
        # doesnt use the rotation


#-----------------

#Idee: Alte Zutände weniger gewichten, also Spur mit 0.9,0.91,...,1
        map = state["map"]
        reward = state["reward"]
        done = state["done"]
        if self.begin:
            self.env_map = copy.copy(map)
            self.begin = False
        player_map = np.zeros(shape=(self.state_cnt[1], self.state_cnt[2]))
        #player_map = copy.copy(self.spur_map)
        player_coords = (state["playerPos"][1],
                         state["playerPos"][0])
        player_map[player_coords] =  math.radians(state["playerRot"]) # change back to 1?????????????????
        #player_map[player_coords] =  1
        #player_map[player_coords] =  np.sin(math.radians(state["playerRot"]))
        c_map = copy.copy(map)
        c_map[player_coords] =5  #5 
        self.spur_map[player_coords] = 1
        #rotrot = die nur rotation map failed!
        if multi_player:
            print(multi_player)
            opponent_map = self.zero_map
            opponent_coords = (
                int(state["opponentPos"][1]), int(state["opponentPos"][0]))
            opponent_map[opponent_coords] = 1
            features = np.array([map, player_map, opponent_map])
        else:
            features = np.array([c_map, player_map])

        return features, reward, done

    def cnn_preprocess_state_2(self, state, multi_player=False):
        # converts given state into fitting state for CNN with only matrices
        # creates a diffMap with only zeros except a 1 in the player position
        # doesnt use the rotation

        player_map = copy.copy(state["map"])
        reward = state["reward"]
        done = state["done"]
        player_coords = (int(state["playerPos"][0]),
                         int(state["playerPos"][1]))
        player_map[player_coords] = 1
        features = np.array([player_map])

        return features, reward, done

""" Preprocessors for LFA """
class LFAPreprocessor:

    def __init__(self, size):
        self.map_size = size

    def lfa_preprocess_state_map(self, state, multi_player=False):

        # converts given state into fitting state for the LFA. Has to be only scalars for the state
        # takes player position as argument and one scalar for each pixel in the environment. The value of theses pixels is max(0, cos(a)/distance
        # where distance is the distance from the pixel to the player and a is
        # the angle between player rotation and vector from position to pixel.
        features = self.get_features(
            state["playerPos"],
            state["playerRot"],
            state["map"])
        if (multi_player):
            opponent_features = self.get_features_tron(
                state["opponentPos"], state["opponentRot"], state["map"])
            features = np.append(features, opponent_features)
        return features, state["reward"], state["done"]

    def get_features_tron(self, pos, rot, map):
        m = copy.copy(map)
        x = pos[0] / self.map_size
        
        y = pos[1] / self.map_size
        # Compute position vector pv

        pv_x = np.cos(rot)

        for i in range(m[0].size):  # y coord
            for j in range(m[0].size):  # x coord
                if(m[i][j] == 1):
                    d_x = np.abs(pos[0]-i) * x
                    d_y = np.abs(pos[1] - j) * y
                    m[i][j] = d_x + d_y

        m = m.reshape(1, -1)[0]
        x = 0.5 - x
        y = 0.5 - y
        abs = np.sqrt((x**2) + (y**2))
        pos = np.array([abs, x, y,pv_x])
        features = np.append(pos, m)

        return features
    
    def get_features(self, pos, rot, map):

        m = copy.copy(map)
        y = pos[0] / self.map_size
        x = pos[1] / self.map_size
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
        abs = np.sqrt((x**2) + (y**2))
        pos = np.array([abs, y,x,pv_x])
        features = np.append(pos, m)

        return features

    def lfa_preprocess_state_feat(self, state, multi_player=False):
        features = []
        
        #features.append(self.basic_features(state)) #0.0001
        #features.append(self.advanced_features(state)) #0.0005
        #features.append(self.count_in_rows(state["map"])) #0.001
        features.append(self.binary_features(state)) #0.001
        features = np.array(features)[0].flatten()
        #print(state["map"])
        
        #features = np.array([abs, dy, dx, np.cos(state["playerRot"])])
        return features, state["reward"], state["done"]

    def basic_features(self,state):
        features = []
        # a)
        p = state["playerPos"]
        dx = p[0]/self.map_size
        dy = p[1]/self.map_size
        # b)
        
        
        features.append(dx)      
        features.append(dy)
        features.append(1-dx)      
        features.append(1-dy)
        # c)
        abs = np.sqrt((dx**2) + (dy**2))
        features.append(abs)

        return np.asarray(features)
        
    def advanced_features(self,state):
        features = []
        p = state["playerPos"]
        
        # distances to walls
        dist_w_r = self.map_size - p[0]-1
        dist_w_l = p[0]-2 #already as pos in
        dist_w_d = self.map_size - p[1]-1
        dist_w_u = p[1]-2 #already as pos in        
        features.append(dist_w_r)
        features.append(dist_w_l)
        features.append(dist_w_d)
        features.append(dist_w_u)

        # distances in directions
        dist_r = self.distance('r',state["map"],state["playerPos"],state["done"])
        dist_l = self.distance('l',state["map"],state["playerPos"],state["done"])
        dist_d = self.distance('d',state["map"],state["playerPos"],state["done"])
        dist_u = self.distance('u',state["map"],state["playerPos"],state["done"])     
        features.append(dist_r)
        features.append(dist_l)
        features.append(dist_d)
        features.append(dist_u)
        features = np.asarray(features)/(self.map_size)
        return features

    def binary_features(self,state):
        
        #Field on your x side full:
        f = np.zeros(8)
        if (state["done"]):
            #print(state["map"])
            return f
        m = state["map"]
        p = state["playerPos"]
        r = state["playerRot"]
        #m[(p[1],p[0])] = 9
        if (m[(p[1],p[0]+1)] == 1): #r
            f[0] = 1        
        if (m[(p[1],p[0]-1)] == 1): #l
            f[1] = 1        
        if (m[(p[1]+1,p[0])] == 1): #d
            f[2] = 1        
        if (m[(p[1]-1,p[0])] == 1): #u
            f[3] = 1
     
        # full Neighbor fields:
        s = np.sum(f)
        if ( s == 0): f[4] = 1
        elif ( s == 1): f[5] = 1
        elif ( s == 2): f[6] = 1
        elif ( s == 3): f[7] = 1
        
        """
        # player rotation
        if ( r == 0): f[8] = 1
        elif ( r == 1): f[9] = 1
        elif ( r == 2): f[10] = 1
        elif ( r == 3): f[11] = 1
                
        
        #diagonal
        if (m[(p[1]+1,p[0]+1)] == 1): #r
            f[4] = 1        
        if (m[(p[1]-1,p[0]-1)] == 1): #l
            f[5] = 1        
        if (m[(p[1]+1,p[0]-1)] == 1): #d
            f[6] = 1        
        if (m[(p[1]-1,p[0]+1)] == 1): #u
            f[7] = 1
        """
        return f
    
    def count_in_rows (self, map):
        # counts the number of full fields in every row and column
        features = []
        full = np.zeros((4,self.map_size-2))
        b1 = True
        b2 = True
        b3 = True
        b4 = True
        for i in range(1,self.map_size-1):  
            
            if ((np.sum(map[self.map_size-i]) == self.map_size) and b1):
                b1 = False
                full[0][i] = 1
            if ((np.sum(map[i]) == self.map_size) and b2):
                b2 = False
                full[1][i] = 1
            if ((np.sum(map[:,self.map_size-i]) == self.map_size) and b3):
                b3 = False
                full[2][i] = 1
            if ((np.sum(map[:,i]) == self.map_size) and b4):
                b4 = False
                full[3][i] = 1


        #print(full)
        features.append(full.flatten())
        #f = np.sum(map) # / ((self.map_size)**2.0)
        #features.append(f)
        return np.asarray(features)
        
        
    def distance(self, dir, curr_map, pos, done):
        if (done): return 0
        # works now perfectly for Tron
        if ( dir == 'r'):            
            x_adder = 1
            y_adder = 0
        elif ( dir == 'l'):            
            x_adder = -1
            y_adder = 0
        elif ( dir == 'u'):            
            x_adder = 0
            y_adder = -1
        elif ( dir == 'd'):            
            x_adder = 0
            y_adder = 1
        distance = 0
        i = 0
        while True:
            # look in one direction until you see a wall
            i += 1
            y = pos[0] + y_adder * i
            x = pos[1] + x_adder * i
            if curr_map[y,x] != 0:
                break
            distance += 1
        return distance        
        
        
        
        
        