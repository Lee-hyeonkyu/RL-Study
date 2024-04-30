import torch
import gymnasium
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


transition = namedtuple(
    "transition", ("state", "action", "next_state", "reward", "done")
)


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

        batch = buffer.sample()
        batch = transition(*zip(*batch))

        batch_size = buffer.batch_size

        states = batch.state
        next_states = batch.next_state
        rewards = torch.FloatTensor(batch.reward)
        dones = torch.FloatTensor(batch.done)
        actions = torch.tensor(batch.action).long()

        z_dist = torch.from_numpy(
            np.array(
                [[self.V_min + i * self.delta_z for i in range(self.N)]] * batch_size
            )
        )
        z_dist = torch.unsqueeze(z_dist, 2).float()

        _, Q_log_dist = self.q.forward(torch.FloatTensor(states))
        Q_log_dist = Q_log_dist[torch.arange(batch_size), actions, :]

        Q_next_target_dist, _ = self.target.forward(torch.FloatTensor(next_states))

        Q_target = torch.matmul(Q_next_target_dist, z_dist).squeeze(1)

        max_Q_next_target = Q_next_target_dist[
            torch.arange(batch_size), torch.argmax(Q_target, dim=1).squeeze(1), :
        ]

        m = torch.zeros(batch_size, self.N)

        for j in range(self.N):
            T_zj = torch.clamp(
                rewards + self.gamma * (1 - dones) * (self.V_min + j * self.delta_z),
                min=self.V_min,
                max=self.V_max,
            )

            bj = (T_zj - self.V_min) / self.delta_z
            l = bj.floor().long()
            u = bj.ceil().long()

            Q_l = torch.zeros(m.size())
            Q_l.scatter_(
                1,
                l.reshape((batch_size, 1)),
                max_Q_next_target[:, j].unsqueeze(1)
                * (u.float() - bj.float()).unsqueeze(1),
            )
            Q_u = torch.zeros(m.size())
            Q_u.scatter_(
                1,
                u.reshape((batch_size, 1)),
                max_Q_next_target[:, j].unsqueeze(1)
                * (bj.float() - l.float()).unsqueeze(1),
            )
            m += Q_l
            m += Q_u

        self.optimizer.zero_grad()
        loss = -torch.sum(torch.sum(torch.mul(Q_log_dist, m), -1), -1) / batch_size

        loss.backward()
        self.optimizer.step()
        self.target_update_frequency += 1

        if self.target_update_frequency == 100:
            self.update_target()
            self.target_update_frequency = 0


def analysis_tool(number_of_atoms):

    EPISODES = 500
    BATCH_SIZE = 64
    ENV = gymnasium.make("CartPole-v1")
    REWARD_BUFFER = deque([0.0], maxlen=100)
    MEMORY_BUFFER = ReplayMemory(10000, BATCH_SIZE)
    REWARDS = list()
    AGENT = Agent(number_of_atoms)
    EPSILON_START = 0.5
    EPSILON_END = 0.0
    EPSILON_DECAY = EPISODES // 2

    for EPISODE in range(EPISODES):
        state = ENV.reset()
        if type(state) is tuple:
            state = state[0]
        epsilon = np.interp(EPISODE, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        print(f"Episode {EPISODE} started. epsilon value is {epsilon}")
        episode_reward = 0

        for EPISODE in range(EPISODES):

            action = AGENT.action(torch.FloatTensor([state]), epsilon)
            next_state, reward, done, _, _ = ENV.step(action)
            MEMORY_BUFFER.push(state, action, next_state, reward, done)
            AGENT.update(MEMORY_BUFFER)
            state = next_state
            episode_reward += reward
            if done:
                REWARD_BUFFER.append(episode_reward)
                REWARDS.append(np.mean(REWARD_BUFFER))

    return REWARDS
