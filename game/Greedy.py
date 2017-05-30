'''
Created on 16.02.2017

@author: Martin
'''
import random
import numpy

""" Heuristischer Suchalgorithmus: """


class Greedy(object):

    def init(self, map_size):

        self.map_size = map_size
        
    def policy(self, action, state):
        map = state["map"]
        pos =  state["playerPos"]
        dist_to_wall = self.distance(action, map, pos)
        if dist_to_wall == 0:
            if action == 0: a1 = 3 
            else: a1 = action-1
            a2 = (action+1)%4
            d1 = self.distance(a1, map, pos)
            d2 = self.distance(a2, map, pos)
            if d1 > d2: return a1
            else: return a2
        else:
            return action
        
    def distance(self, dir, curr_map, pos):
        # works now perfectly for Tron
        if ( dir == 0):            
            x_adder = 1
            y_adder = 0
        elif ( dir == 2):          #rlud  
            x_adder = -1
            y_adder = 0
        elif ( dir == 1):            
            x_adder = 0
            y_adder = -1
        elif ( dir == 3):            
            x_adder = 0
            y_adder = 1
        distance = 0
        i = 0
        while True:
            # look in one direction until you see a wall
            i += 1
            y = pos[1] + y_adder * i
            x = pos[0] + x_adder * i
            if curr_map[y,x] != 0:
                break
            distance += 1
        return distance        
        

    