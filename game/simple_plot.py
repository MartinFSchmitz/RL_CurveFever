import matplotlib.pyplot as plt
import numpy as np
import pickle

""" Class for creating Plots from given reward Arrays """

def make_plot(x, name, step):
    """ creates and saves plot of given arrays of rewards
    Input:
    x: list of arrays with values
    name: name of directory and file name
    step: steps of the regression curve 
    Output:
    Saved several Plots"""
    
        # Example data
    labels = ('Q-Learning', 'DQN', 'REINFORCE-LA', 'REINFORCE-CNN', 'A3C' , 'Greedy')
    
    leng = x[0].size
    # ToDO cut all
    step_x = []
    all_arrays = []

    rewards = 0

    for j in range(len(x)):
        rewards = 0
        for i in range(leng):  # xrange
            rewards += x[j][i]
            if i % step == 0 and i != 0:
                step_x.append(rewards / step)
                rewards = 0

        reward_step_array = np.asarray(step_x)
        all_arrays.append(reward_step_array)
        step_x = []

    episodes = np.arange(0, leng, 1)
    episodes_step = np.arange(step / 2, leng - step / 2, step)

    #plt.plot( episodes,reward_array,linewidth=0.2,color='go')
    for j in range(len(x)):
        plt.plot(episodes_step, all_arrays[j], linewidth=1, label=labels[j])
    plt.plot(
        episodes_step,
        np.zeros(
            len(episodes_step)),
        linewidth=0.01,
        color='black')  # graph to show x axis
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # Where to put legend
    plt.legend(bbox_to_anchor=(0.75, 0.8), loc=2, borderaxespad=0.)
    #plt.title(name.upper() + ': Rewards per Episode')
    plt.title(name + ': Rewards per Episode')
    plt.grid(True)
    plt.savefig("data/plots/" + name + "_plot.png")
    print("...made plot")
    print("------Finished------")
    plt.show()
#------------------------------------------------------------------------
def make_bar_plot(x, name, number):

    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    # Example data
    names = ('Basic', 'Advanced', 'Saturation')
    if(len(x)!=len(names)):
        print("Error: Change Names or Data")
    y_pos = np.arange(len(x))
    leng = x[0].size    
    values = np.zeros(len(x))
    std_der = np.zeros(len(x))
    for j in range(len(x)):
        rewards = 0
        for i in range(leng-number, leng):  # xrange
            rewards += x[j][i]
        a = x[j][leng-number:]
        values[j] = np.mean(a)
        std_der[j] = np.std(a)

    error = 5
    
    ax.barh(y_pos, values, xerr=std_der, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Reward')
    ax.set_title('Average Reward over last ' + str(number) + ' games')
    plt.savefig("data/plots/" + name + "bar_plot.png")
    print("...made plot")
    print("------Finished------")
    plt.show()
#------------------------------------------------------------------------
print("-----Plot Maker-----")
data = []


with open('data/plots/exp_1/lfa.p', 'rb') as pickle_file:
    rewards_1 = pickle.load(pickle_file)
with open('data/plots/exp_1/dqn.p', 'rb') as pickle_file:
    rewards_2 = pickle.load(pickle_file)
with open('data/plots/exp_1/lfa_rei.p', 'rb') as pickle_file:
    rewards_3 = pickle.load(pickle_file)
with open('data/plots/exp_1/rei.p', 'rb') as pickle_file:
    rewards_4 = pickle.load(pickle_file)
with open('data/plots/exp_1/a3c.p', 'rb') as pickle_file:
    rewards_5 = pickle.load(pickle_file)
    rewards_5 = rewards_5[:50000]
rewards_6 = np.zeros(50000) + 700
#rewards_6.tolist()
print(rewards_6)
data.append(rewards_1)
data.append(rewards_2)
data.append(rewards_3)
data.append(rewards_4)
data.append(rewards_5)
data.append(rewards_6)

make_plot(data, 'Experiment 1', 100)
#make_bar_plot(data, 'LFA-Preprozessoren', 1000)








