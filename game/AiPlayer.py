'''
Created on 21.11.2016

@author: Martin
'''



from Player import Player
import random
import numpy




""" Heuristischer Suchalgorithmus: """
class AiPlayer(Player):
         
    def doAction(self, action, map, diffmap): # Fahre gerade aus, bis Abstand zu Wand < epsilon / In jedem Zeitschritt
       
        action = 0
        epsilon = 2000 # epsilon >= 3 | ToDo: try epsilon --> infinity (= no epsilon)
        dist_to_wall = self.distance(self.rotation, map)  
        if dist_to_wall <= epsilon : 
            #action = self.maxdist_policy(map)
            action = self.not_mindist_policy(map)  
        self.rotate = action
                      
                
                      
    def distance(self, degree, curr_map): # return the distance to the next wall in given angle
        #a = self.lookUp(degree)
        x, y = self.init_lookUp()
        degree = int(degree)
        if degree >= 360 : degree -= 360
        if degree < 0 : degree += 360
        x_adder = x[degree] # use the angle to define a path to look at
        y_adder = y[degree]
        #print("x: " + str(x_adder))
        #print("y: " + str(y_adder))
        curr_field = (int(self.x),int(self.y))
        last_field = curr_field
        distance = 0
        i = 0
        while True: # look in one direction until you see a wall
            i += 1
            x = int(self.x + x_adder * i)
            y = int(self.y + y_adder * i)
            
            curr_field = (x,y)
            if (curr_field != last_field): 
                distance += 1
                last_field = curr_field
            if ((self.outOfMap(curr_field) or curr_map[curr_field[0],curr_field[1]]) and distance > 2) != 0: break

        return distance 
        
    def outOfMap (self, curr_field):

        return curr_field[0] < 0 or curr_field[0] > self.mapSize[0] or curr_field[1] < 0 or curr_field[1] > self.mapSize[1]

    def lookUp (self, degree):
        # oh wow this method is sooooo beautiful, compare to sin and cos
        #a = ((0,-1),(-0.5,-0.5),(-1, 0),(-0.5,0.5),(0,1),(0.5,0.5),(1,0),(0.5,-0.5))
        y = ( 0, -0.1, -0.2, -0.3, -0.4 ,-0.5, -0.6, -0.7, -0.8, -0.9, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
        x =(  -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4 ,-0.5, -0.6, -0.7, -0.8, -0.9)
        if(degree>0 and degree < 360-(360/len(x))): split = int(degree/360*(len(x)))
        else: split = len(x)-1
        return (-x[split],-y[split])
    
    def init_lookUp (self):
        
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
    
    def maxdist_policy(self, map):  #biege in bessere Richtung ab. ( besser ermittelbar durch: argmax oder durchschnitt der Abstaende) 
    
        deg_range = 1 # the degree difference between 2 distance calls
        
        max_f = self.distance(self.rotation , map)
        max_l = 0
        max_r = 0
        for angle in range(1, 79/ deg_range):
            dist_l = self.distance(self.rotation - (angle*deg_range) , map)              
            dist_r = self.distance(self.rotation + (angle*deg_range) , map)
            if(dist_l > max_l): max_l = dist_l
            if(dist_r > max_r): max_r = dist_r
        
        if(max_f >= max_l and max_f >= max_r): action = 0 # 0:=forward, 1:=turn right, -1:=turn left
        elif(max_l > max_r): action = -1
        else: action = 1     

        return action
   
    def not_mindist_policy (self, map): #biege in bessere Richtung ab. ( besser ermittelbar durch: nicht argmin oder durchschnitt der Abstaende)
    
        deg_range = 1 # the degree difference between 2 distance calls
        
        min_f = self.distance(self.rotation , map)
        min_l = 1000
        min_r = 1000
        for angle in range(1, 44/ deg_range):
            dist_l = self.distance(self.rotation - (angle*deg_range) , map)              
            dist_r = self.distance(self.rotation + (angle*deg_range) , map)
            if(dist_l < min_l): min_l = dist_l
            if(dist_r < min_r): min_r = dist_r
        
        #if(min_f >= min_l and min_f >= min_r): action = 0 # 0:=forward, 1:=turn right, -1:=turn left
        if(min_l > min_r): action = -1
        else: action = 1     

        return action






        self.rotate = action
        
        
