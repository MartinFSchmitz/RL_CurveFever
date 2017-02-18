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
        self.look_up = self.init_look_up()
                
                      
    def distance(self, degree, curr_map, pos): # return the distance to the next wall in given angle
        x, y = self.look_up
        degree = int(degree)
        if degree >= 360 : degree -= 360
        if degree < 0 : degree += 360
        x_adder = x[degree] # use the angle to define a path to look at
        y_adder = y[degree]
        #print("x: " + str(x_adder))
        #print("y: " + str(y_adder))
        curr_field = (int(pos[0]),int(pos[1]))
        last_field = curr_field
        distance = 0
        i = 0
        while True: # look in one direction until you see a wall
            i += 1
            x = int(pos[0] + x_adder * i)
            y = int(pos[1] + y_adder * i)
            
            curr_field = (x,y)
            if (curr_field != last_field): 
                distance += 1
                last_field = curr_field
            if ((self.outOfMap(curr_field) or curr_map[curr_field[0],curr_field[1]]) and distance > 2) != 0: break

        return distance 
        
    def outOfMap (self, curr_field):

        return curr_field[0] < 0 or curr_field[0] > self.map_size or curr_field[1] < 0 or curr_field[1] > self.map_size
    
    def init_look_up (self):
        
        x = [None]*360
        y = [None]*360
        for i in range(0, 90):
            a = i * (1.0/90.0)
            y[i]= a
            y[i+90] = (1.0-a)
            y[i+180] = -a
            y[i+270] = -(1.0-a)
            
            x[i]= (1.0-a)
            x[i+90] = -a
            x[i+180] = -(1.0-a)
            x[i+270] = a

        return x, y    
    
    def maxdist_policy(self, map, pos, rotation):  #biege in bessere Richtung ab. ( besser ermittelbar durch: argmax oder durchschnitt der Abstaende) 
    
        deg_range = 1 # the degree difference between 2 distance calls
        
        max_f = self.distance(rotation , map, pos)
        max_l = 0
        max_r = 0
        for angle in range(1, 79/ deg_range):
            dist_l = self.distance(rotation - (angle*deg_range) , map, pos)              
            dist_r = self.distance(rotation + (angle*deg_range) , map, pos)
            if(dist_l > max_l): max_l = dist_l
            if(dist_r > max_r): max_r = dist_r
        
        if(max_f >= max_l and max_f >= max_r): action = 0 # 0:=forward, 1:=turn right, -1:=turn left
        elif(max_l > max_r): action = -1
        else: action = 1     

        return action
   
    def not_mindist_policy (self, map, pos, rotation): #biege in bessere Richtung ab. ( besser ermittelbar durch: nicht argmin oder durchschnitt der Abstaende)
    
        deg_range = 1 # the degree difference between 2 distance calls
        
        min_f = self.distance(rotation , map, pos)
        min_l = 1000
        min_r = 1000
        for angle in range(1, 44/ deg_range):
            dist_l = self.distance(rotation - (angle*deg_range) , map, pos)              
            dist_r = self.distance(rotation + (angle*deg_range) , map,  pos)
            if(dist_l < min_l): min_l = dist_l
            if(dist_r < min_r): min_r = dist_r
        
        #if(min_f >= min_l and min_f >= min_r): action = 0 # 0:=forward, 1:=turn right, -1:=turn left
        if(min_l > min_r): action = -1
        else: action = 1     

        return action
     
