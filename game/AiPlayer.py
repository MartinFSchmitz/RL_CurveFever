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
        epsilon = 20 # epsilon >= 3
        dist_to_wall = self.distance(self.rotation, map)    
        #print(dist_to_wall)    
        if dist_to_wall <= epsilon : action = self.avoidWall(map)    
        self.rotate = action
                      
    def distance(self, degree, curr_map): # return the distance to the next wall in given angle
        a = self.lookUp(degree)
        x_adder = a[0] # use the angle to define a path to look at
        y_adder = a[1]
        print (a)

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
    
    def avoidWall(self, map):  #biege in bessere Richtung ab. ( besser ermittelbar durch: argmax oder durchschnitt der Abstaende) 
    
        angle = 45
        l = self.distance(self.rotation - angle , map)
        r = self.distance(self.rotation + angle , map)
        
        if( l > r): action = -1 # 1 := turn right, -1 := turn left
        else: action = 1
             
        return action
   

    







        self.rotate = action
        
        
