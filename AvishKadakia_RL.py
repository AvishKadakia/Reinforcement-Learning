#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop


# ## Defining Parameters

# In[2]:


#Training parameters
n_episodes = 300
n_win_ticks = 200

gamma = 1.0 # Discount Factor
epsilon = 1.0 # Exploration Factor
epsilon_decay = 0.99
epsilon_min = 0.01
lr = 0.01 #learning rate
lr_decay = 0.01

batch_size = 64 # how may samples to train on from memmory
monitor = False
quiet = False


# ## Setting up the Cart Pole environment

# In[3]:


# Environment Parameter
memory = deque(maxlen=10000)
env = gym.make('CartPole-v0')
env.max_episode_steps = 500
input_shape = 4
action_space = 2


# ## Neural Network Architechture

# In[4]:


def OurModel(input_shape, action_space):
    # Input Layer of state size(4)
    X_input = Input(input_shape)
    # Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu")(X_input)
    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu")(X)
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu")(X)
    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear")(X)
    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model


# ## Agent

# In[5]:


class DQNAgent:
    def __init__(self):
        #Setting Up environment and initialising parameters
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000
        # creating main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory and then taining neural network on the experience
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # Updating Q value for the action
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def run(self):
        flag = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                #self.env.render()
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    self.remember(state, action, reward, next_state, done)
                else:
                    self.remember(state, action, -100, next_state, done)
                
                state = next_state
                i += reward
                if done:       
                               
                    print(f"episode: {e}/{self.EPISODES}, score: {i}, e: {self.epsilon}")
                    if i >= 200:
                        print("|----------------------------------Solved----------------------------------|")
                        print(f"episode: {e}/{self.EPISODES}, score: {i}, e: {self.epsilon}")
                        flag = 1
                        break
                    if flag == 1:
                      break
                if flag == 1:
                      break
                self.replay()
            if flag == 1:
                break


# ## Executing Model

# In[6]:


print("----------Method 1 Using DQN-------")
agent = DQNAgent()
agent.run()


# ## Imports

# In[12]:


import gym
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randint
from statistics import median, mean
np.random.seed(seed=20)


# ## Settingup Initial Parameters

# In[13]:


award_set =[]
test_run = 15
best_gen =[]
n_of_generations = 1000


# ## Setting Up Environment

# In[14]:


env = gym.make('CartPole-v1')

ind = env.observation_space.shape[0]
adim = env.action_space.n #discrete


# ## Creating Neural Network

# In[15]:


def softmax(x):
    x = np.exp(x)/np.sum(np.exp(x))
    return x

def lreLu(x):
    alpha=0.2
    return tf.nn.relu(x)-alpha*tf.nn.relu(-x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def reLu(x):
    return np.maximum(0,x)

def nn(obs,in_w,in_b,hid_w,out_w):

    obs = obs/max(np.max(np.linalg.norm(obs)),1) 

    Ain = reLu(np.dot(obs,in_w)+in_b.T)

    Ahid = reLu(np.dot(Ain,hid_w))
    lhid = np.dot(Ahid,out_w)

    out_put = reLu(lhid)
    out_put = softmax(out_put)
    out_put = out_put.argsort().reshape(1,adim)
    act = out_put[0][0] #index of discrete action

    return act


# ## Generate initial set of weights and bias

# In[17]:


def intial_gen(test_run):
    input_weight = []
    input_bias = []

    hidden_weight = []
    out_weight = [] 

    in_node = 4  
    hid_node = 2

    for i in range(test_run):
        in_w = np.random.rand(ind,in_node)
        input_weight.append(in_w)

        in_b = np.random.rand((in_node))
        input_bias.append(in_b)

        hid_w = np.random.rand(in_node,hid_node)
        hidden_weight.append(hid_w)


        out_w = np.random.rand(hid_node, adim)
        out_weight.append(out_w)

    generation = [input_weight, input_bias, hidden_weight, out_weight]
    return generation


# ## Run environment randomly 

# In[18]:



def rand_run(env,test_run):
    award_set = []
    generations = intial_gen(test_run)

    for episode in range(test_run):# run env 10 time
        in_w  = generations[0][episode]
        in_b = generations[1][episode]
        hid_w =  generations[2][episode]
        out_w =  generations[3][episode]
        award = run_env(env,in_w,in_b,hid_w,out_w)
        award_set = np.append(award_set,award)
    gen_award = [generations, award_set]
    return gen_award 


# ## Genetic Algorithm

# In[19]:


def run_env(env,in_w,in_b,hid_w,out_w):
    obs = env.reset()
    award = 0
    for t in range(300):
        #env.render() this slows the process theredore commented
        action = nn(obs,in_w,in_b,hid_w,out_w)
        obs, reward, done, info = env.step(action)
        award += reward 
        if done:
            break
    return award

def mutation(new_dna):

    j = np.random.randint(0,len(new_dna))
    if ( 0 <j < 10): # controlling rate for amount of mutation
        for ix in range(j):
            n = np.random.randint(0,len(new_dna)) #random postion for mutation
            new_dna[n] = new_dna[n] + np.random.rand()

    mut_dna = new_dna

    return mut_dna

def crossover(Dna_list):
    newDNA_list = []
    newDNA_list.append(Dna_list[0])
    newDNA_list.append(Dna_list[1]) 

    for l in range(10):  # generation after crassover
        j = np.random.randint(0,len(Dna_list[0]))
        new_dna = np.append(Dna_list[0][:j], Dna_list[1][j:])

        mut_dna = mutation(new_dna)
        newDNA_list.append(mut_dna)

    return newDNA_list

#Generate new set of weights and bias from the best previous weights and bias

def reproduce(award_set, generations):

    good_award_idx = award_set.argsort()[-2:][::-1] # here only best 2 are selected 
    good_generation = []
    DNA_list = []

    new_input_weight = []
    new_input_bias = []

    new_hidden_weight = []

    new_output_weight =[]

    new_award_set = []


    #Extraction of all weight info into a single sequence
    for index in good_award_idx:

        w1 = generations[0][index]
        dna_in_w = w1.reshape(w1.shape[1],-1)

        b1 = generations[1][index]
        dna_b1 = np.append(dna_in_w, b1)

        w2 = generations[2][index]
        dna_whid = w2.reshape(w2.shape[1],-1)
        dna_w2 = np.append(dna_b1,dna_whid)

        wh = generations[3][index]
        dna = np.append(dna_w2, wh)


        DNA_list.append(dna) # make 2 dna for good gerneration

    newDNA_list = crossover(DNA_list)

    for newdna in newDNA_list: # collection of weights from dna info

        newdna_in_w1 = np.array(newdna[:generations[0][0].size]) 
        new_in_w = np.reshape(newdna_in_w1, (-1,generations[0][0].shape[1]))
        new_input_weight.append(new_in_w)

        new_in_b = np.array([newdna[newdna_in_w1.size:newdna_in_w1.size+generations[1][0].size]]).T #bias
        new_input_bias.append(new_in_b)

        sh = newdna_in_w1.size + new_in_b.size
        newdna_in_w2 = np.array([newdna[sh:sh+generations[2][0].size]])
        new_hid_w = np.reshape(newdna_in_w2, (-1,generations[2][0].shape[1]))
        new_hidden_weight.append(new_hid_w)

        sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
        new_out_w   = np.array([newdna[sl:]]).T
        new_out_w = np.reshape(new_out_w, (-1,generations[3][0].shape[1]))
        new_output_weight.append(new_out_w)

        new_award = run_env(env, new_in_w, new_in_b, new_hid_w, new_out_w) #bias
        new_award_set = np.append(new_award_set,new_award)

    new_generation = [new_input_weight,new_input_bias,new_hidden_weight,new_output_weight]

    return new_generation, new_award_set


def evolution(env,test_run,n_of_generations):
    gen_award = rand_run(env, test_run)
    current_gens = gen_award[0] 
    current_award_set = gen_award[1]
    best_gen =[]
    A =[]
    for n in range(n_of_generations):
        new_generation, new_award_set = reproduce(current_award_set, current_gens)
        current_gens = new_generation
        current_award_set = new_award_set
        avg = np.average(current_award_set)
        a = np.amax(current_award_set)
        print(f"generation: {n+1}, score: {a}")
        if np.amax(current_award_set) >= 200:
            print("|----------------------------------Solved----------------------------------|")
            print(f"generation: {n}/{n_of_generations}, score: {np.amax(current_award_set)}")
            break
        
        A = np.append(A, a)

    Best_award = np.amax(A)
    



# ## Executing Model

# In[23]:


print("----------Method 2 Using NN with Genetic Algorithm-------")

evolution(env, test_run, n_of_generations)

