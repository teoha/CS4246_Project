import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 2000
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20

# Compute loss by comparing the target network and model network
# We have a set (state, action) pair

def compute_loss(model, states, q_values):
    # Q function generated by apprentice model
    input = model(states)
    # Q function generated by target model
    target_loss = q_values
    huberloss = nn.SmoothL1Loss()
    loss = huberloss(input, target_loss)
    return loss

def optimize(model, D, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    D: (state, q_values) tuple
    '''
    dt = np.dtype('object, object')
    d = np.array(D, dtype=dt)
    d.dtype.names = ['state', 'q_values']
    states = np.stack(d['state'])
    q_values = np.stack(d['q_values'])
    batch = (states, q_values)

    loss = compute_loss(model, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model

def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`. 
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)