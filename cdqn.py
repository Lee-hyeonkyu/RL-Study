import torch
import gym
import torch.nn as nn
import numpy as np
import torch.functional as F
from collections import namedtuple
import math
from collections import deque
import random
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * N),
        )
        self.N = N

    def forward(self, state):
        x = self.net(state)
        return nn.Softmax(dim=2)(x.view(-1, 2, self.N)), nn.LogSoftmax(dim=2)(
            x.view(-1, 2, self.N)
        )


transition = namedtuple("transition", ("state", "action", "next_state"))


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
