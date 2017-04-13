import matplotlib.pyplot as plt
import numpy as np
import pickle


def make_plot(x, name, step):
    leng = x[0].size
    #ToDO cut all
    step_x = []
    all_arrays = []
    rewards = 0

    for j in range (len(x)):
        for i in range (leng): #xrange
            rewards += x[j][i]
            if i % step == 0 and i != 0:
                step_x.append(rewards/step)
                rewards = 0
                
        reward_step_array = np.asarray(step_x)   
        all_arrays.append(reward_step_array)
        step_x = []
        
        
    #episodes = np.arange(0, leng, 1)   
    episodes_step = np.arange(step/2, leng - step/2 , step)    
     
    #plt.plot( episodes,reward_array,linewidth=0.2,color='g')
    for j in range (len(x)):
        plt.plot(episodes_step,all_arrays[j],linewidth=1)
    plt.plot(episodes_step,np.zeros(len(episodes_step)),linewidth=0.01,color = 'black') # graph to show x axis
    plt.xlabel('Number of episode')
    plt.ylabel('Reward')
    plt.title(name.upper() + ': Rewards per episode')
    plt.grid(True)
    plt.savefig("data/plots/" + name +"_plot.png")
    print("...made plot")
    print("------Finished------")
#------------------------------------------------------------------------

print("-----Plot Maker-----")
data = []
    
with open('data/plots/save.p','rb') as pickle_file:
    rewards = pickle.load(pickle_file)
with open('data/plots/rewards.p','rb') as pickle_file:
    rewards_2 = pickle.load(pickle_file)

data.append(rewards)
data.append(rewards_2)
make_plot(data,'test', 100)

    
    
    
    
    
    
    
    