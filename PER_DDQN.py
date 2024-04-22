import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from collections import deque
import gymnasium

import warnings

warnings.filterwarnings("ignore")


torch.manual_seed(4444)
np.random.seed(4444)


class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tanh(), nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self.forward(state_t)
        max_q_values = torch.argmax(state_t)
        action = max_q_values.detach().item()
        return action


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedBuffer:

    def __init__(self, max_size):
        self.sum_tree = SumTree(max_size)
        self.current_length = 0

    def push(self, state, action, reward, next_state, done):

        priority = 1.0 if self.current_length == 0 else self.sum_tree.tree.max()
        self.current_length = self.current_length + 1

        experience = (state, action, np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience)

    def sample(self, batch_size):
        batch_idx, batch, priorities = [], [], []
        segment = self.sum_tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.sum_tree.get(s)
            priorities.append(p)
            # print(data)
            batch.append(data)
            batch_idx.append(idx)

        sampling_probabilites = priorities / (self.sum_tree.total() + EP)
        weights_ = np.power(self.sum_tree.n_entries * sampling_probabilites, -BETA)
        weights_ /= weights_.max()
        # print(batch[0])
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (
            (state_batch, action_batch, reward_batch, next_state_batch, done_batch),
            batch_idx,
            weights_,
        )

    def update_priority(self, idx, td_error):
        priority = (np.abs(td_error) + EP) ** ALPHA
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length


BATCH_SIZE = 32
BUFFER_SIZE = 10000
REPLAY_BUFFER = PrioritizedBuffer(BUFFER_SIZE)
LEARNING_RATE = 5e-4
GAMMA = 0.99
MAX_EP = 25000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 20000


ALPHA = 0.7

BETA = 0
BETA_START = 0.5
BETA_END = 1.0
BETA_ANNEAL = 20000


EP = 1e-6  # division by 0 방지

EPISODE_REWARD = 0.0
REWARD_BUFFER = deque([0.0], maxlen=100)
MIN_REPLAY_SIZE = 100

env = gymnasium.make("CartPole-v1")

online_net = DQN(env)
target_net = DQN(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = env.reset()

step = 0

while step != MAX_EP:
    if type(state) is tuple:
        state = state[0]
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    BETA = np.interp(step, [0, BETA_ANNEAL], [BETA_START, BETA_END])

    random_sample = random.random()

    if random_sample > epsilon:
        action = online_net.act(state)
    else:
        action = env.action_space.sample()

    next_state, reward, done, _, _ = env.step(action)
    REPLAY_BUFFER.push(state, action, reward, next_state, done)

    state = next_state
    EPISODE_REWARD += reward

    if done:
        state = env.reset()
        if type(state) is tuple:
            state = state[0]
        REWARD_BUFFER.append(EPISODE_REWARD)
        EPISODE_REWARD = 0.0

    if len(REPLAY_BUFFER) > BATCH_SIZE:

        transitions, idxs, weights_ = REPLAY_BUFFER.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = transitions

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        weights_ = torch.FloatTensor(np.array(weights_)).to(device)

        states = torch.reshape(states, (BATCH_SIZE, 4))
        actions = torch.reshape(actions, (BATCH_SIZE, 1))
        rewards = torch.reshape(rewards, (BATCH_SIZE, 1))
        next_states = torch.reshape(next_states, (BATCH_SIZE, 4))
        dones = torch.reshape(dones, (BATCH_SIZE, 1))
        weights_ = torch.reshape(weights_, (BATCH_SIZE, 1))

        online_with_new_states = online_net.forward(next_states)
        argmax_online_with_new_states = online_with_new_states.argmax(
            dim=1, keepdim=True
        )

        offline_with_new_states = target_net.forward(next_states)
        target_q_vals = torch.gather(
            input=offline_with_new_states, dim=1, index=argmax_online_with_new_states
        )
        targets = rewards + GAMMA * (1 - dones) * target_q_vals

        q_values = online_net.forward(states)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions)
        errors = torch.abs(action_q_values - targets).data.numpy()

        td_errors = torch.pow(action_q_values - targets, 2) * weights_
        td_errors_mean = td_errors.mean()

        optimizer.zero_grad()
        td_errors_mean.backward()
        optimizer.step()

        for idx, error in zip(idxs, errors):
            REPLAY_BUFFER.update_priority(idx, error)

        if step % 1000 == 0:
            target_net.load_state_dict(online_net.state_dict())
            print()
            print("Step", step)
            print("Avg Reward", np.mean(REWARD_BUFFER))
            print("Loss", td_errors_mean)
            print("BETA", BETA)
            print("ALPHA", ALPHA)

    step += 1
