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


class Agent:
    def __init__(self, N):
        self.q = Network(N)
        self.target = Network(N)
        self.update_target()
        self.V_min = -10
        self.V_max = 10
        self.gamma = 0.99
        self.delta_z = (self.V_max - self.V_min) / (N - 1)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.N = N
        self.target_update_frequency = 10

    def action(self, state, epsilon):
        if np.random.randn() < epsilon:
            return np.random.randint(0, 2)

        else:
            z_distribution = torch.from_numpy(
                np.array([[self.V_min + i * self.delta_z for i in range(self.N)]])
            )
            z_distribution = torch.unsqueeze(z_distribution, 2).float()

            Q_dist, _ = self.q.forward(state)
            Q_dist = Q_dist.detach()
            Q_target = torch.matmul(Q_dist, z_distribution)

            return Q_target.argmax(dim=1)[0].detach().cpu().numpy()[0]

        def update_target(self):
            self.target.load_state_dict(self.q.state_dict())

        def update(self, buffer):
            if len(buffer) < buffer.batch_size:
                return

            batch = buffer
