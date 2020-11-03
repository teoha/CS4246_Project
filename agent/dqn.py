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


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        samples = random.sample(self.buffer, batch_size)
        res = [torch.tensor([sample.state for sample in samples],dtype=torch.float,device=device),
               torch.tensor([sample.action for sample in samples]),
               torch.tensor([sample.reward for sample in samples],dtype=torch.float,device=device),
               torch.tensor([sample.next_state for sample in samples],dtype=torch.float,device=device),
               torch.tensor([sample.done for sample in samples],dtype=torch.float,device=device)]
        
        return tuple(res)

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


def compute_loss(model, target, states, actions, rewards, next_states, dones):
    model_actions_qs = model(states) # n x a
    target_actions_qs = target(next_states) # n x a
    model_q = model_actions_qs.gather(1,actions) # n x 1
    not_dones = dones.clone()
    not_dones[not_dones==0]=-1
    not_dones[not_dones==1]=0
    not_dones[not_dones==-1]=1 # n x 1

    target_q = torch.squeeze(rewards) + gamma * torch.max(target_actions_qs,1).values * torch.squeeze(not_dones)
    delta = torch.abs(torch.squeeze(model_q)-target_q)
    delta_1 = 0.5*delta*delta
    delta_2 = delta-0.5
    z = torch.where(delta<1, delta_1, delta_2)

    return torch.mean(z)

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

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

def get_env():
    '''
    Get the sample test cases for training and testing.
    '''
    config = {  'observation_type': 'tensor', 'agent_speed_range': [-2, -1], 'stochasticity': 0.0, 'width': 10,
                'lanes': [
                    LaneSpec(cars=3, speed_range=[-2, -1]), 
                    LaneSpec(cars=4, speed_range=[-2, -1]), 
                    LaneSpec(cars=2, speed_range=[-1, -1]), 
                    LaneSpec(cars=2, speed_range=[-3, -1])
                ] }
    return gym.make('GridDriving-v0', **config)