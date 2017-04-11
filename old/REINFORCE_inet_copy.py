""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

import itertools
#import matplotlib
import numpy as np
import sys
import collections
import pygame
from CurveFever import Learn_SinglePlayer

from Preprocessor import CNNPreprocessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RL_Algo import Brain
import RL_Algo

# This policy gradient implementation is an adaptation of Karpathy's GIST
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

class PG:
    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        r = r.flatten()
        for t in reversed(xrange(0, r.size)):

            running_add = running_add * self.discount + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def train(self, ipy_clear=False, max_episodes=100000000, max_pathlength=200):

        rewards = [] # right type?
        observation = self.env.reset()
        last_state = None # used in computing the difference frame
        xs,hs,dlogps,drs = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        while True:
            
            if self.render: self.env.render()

                # preprocess the observation, set input to network to be difference image
            if not self.preprocessor ==None:
                state = self.preprocessor(observation)
            else:
                state = observation

            x = state - last_state if last_state is not None else np.zeros(self.input_dim, dtype='float32')
            x = x.flatten()
            last_state = state

            # forward the policy network and sample an action from the returned probability
            action_prob = self.model.predict(x.reshape([1,self.input_dim]), batch_size=1).flatten()
            action = np.random.choice( self.env.action_space.n, 1, p=action_prob/np.sum(action_prob) )[0]
            
            # record various intermediates (needed later for backprop)
            xs.append(x) # observation
            
            # Harsh Grad ...
            y = np.zeros([self.env.action_space.n])
            y[action] = 1
            
            # Subtle Grad ...
            #          y = action_prob*0.9
            #          y[action] = action_prob[action] * 1.1
            
            dlogps.append(y) # grad that encourages the action that was tak
            #dlogps.append(y - action_prob) # grad that encourages the action that was tak
            observation, reward, done, info = self.env.step(action)
            reward_sum += float(reward)
            
            drs.append(float(reward)) # record reward (has to be done after we call step() to get reward for previous action)

            if done: # an episode finished
                episode_number += 1

                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs) # states
                epdlogp = np.vstack(dlogps) # y arrays
                epr = np.vstack(drs) # rewards
                xs,hs,dlogps,drs = [],[],[],[] # reset array memory
    
                # compute the discounted reward backwards through time
                discounted_epr = self.discount_rewards(epr)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
    
                epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    
                self.model.fit(epx, epdlogp,
                        nb_epoch=1, verbose=2, shuffle=True)
    
                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                rewards.add(reward_sum)
                print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
                if episode_number % 100 == 0:
                    self.save()
                reward_sum = 0
                observation = self.env.reset() # reset env
                last_state = None
    
                if(self.enable_plots):
                    plt.figure(1)
                    #plt.plot(rewards)
                    rewards.plot()
                    plt.show(block=False)
                    plt.draw()
                    plt.pause(0.001)

            print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')