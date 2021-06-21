import ray
import argparse
import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()

if __name__ == '__main__':
    PATH = './models/model3.pt'
    model = Qnet()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    env = gym.make('CartPole-v1')
    obs = env.reset()
    done = False
    while done:
        env.render()
        act = model(obs).argmax().item()
        nob, rew, done, info = env.step(act)
        obs = nob
