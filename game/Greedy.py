'''
Created on 16.02.2017

@author: Martin
'''
import random
import numpy

""" Greedy algorithm for Tron """


class Greedy(object):

    def init(self, map_size):
        """ init needed parameter for algorithm """
        self.map_size = map_size
        
    def policy(self, action, state):
        """ the policy is a simple greedy strategy waiting until a wall is directly in front of the agent.
        then messure the distances in both sides and move to the side with the higher distance to the agent.
        Go in this direction until another wall is in front of you.
        """
        map = state["map"]
        pos =  state["playerPos"]
        # checks if a wall is directly in fornt of the agent
        dist_to_wall = self.distance(action, map, pos)
        if dist_to_wall == 0:
            # get the two possible actions a1 and a2 in this situation
            if action == 0: a1 = 3 
            else: a1 = action-1
            a2 = (action+1)%4
            # measure distances to wall
            d1 = self.distance(a1, map, pos)
            d2 = self.distance(a2, map, pos)
            # decide which action to take reffering to the measured distance
            if d1 > d2: return a1
            else: return a2
        else:
            return action
        
    def distance(self, dir, curr_map, pos):
        """
        Input: dir: in which direction to look,
               curr_map: current state of the map
               pos: player position/ position to start searching
        Output: the distance from the agent to the next wall in given direction """
        if ( dir == 0): # right            
            x_adder = 1
            y_adder = 0
        elif ( dir == 2): # left          
            x_adder = -1
            y_adder = 0
        elif ( dir == 1): # up            
            x_adder = 0
            y_adder = -1
        elif ( dir == 3): #down            
            x_adder = 0
            y_adder = 1
        distance = 0
        i = 0
        while True:
            # look in one direction until you see a wall
            i += 1
            y = pos[1] + y_adder * i
            x = pos[0] + x_adder * i
            if curr_map[y,x] != 0: # check if you found a wall or used field
                break
            distance += 1
        return distance        
        

    