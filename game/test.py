'''
Created on Feb 26, 2017

@author: marti
'''
import numpy as np
if __name__ == '__main__':
    
    
    
    a = np.arange(4).reshape(2,2)
    a = np.rot90(a,3)
    print(a)