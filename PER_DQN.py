import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import random
from collections import deque
import gymnasium


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, state):
        q_values = self.net(state)
        return q_values


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

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

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedBuffer:

    def __init__(self, max_size, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.current_length = 0
        self.alpha = alpha
        self.beta = beta

    def push(self, state, action, reward, next_state, done):
        priority = 1.0 if self.current_length == 0 else self.sum_tree.tree.max()
        self.current_length += 1
        experience = (state, action, np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience)

    def sample(self, batch_size):
        batch_idx, batch, weights_ = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)

            batch_idx.append(idx)
            batch.append(data)
            prob = p / p_sum
            weight_ = (self.sum_tree.total() * prob) ** (-self.beta)
            weights_.append(weight_)

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
        priority = td_error**self.alpha
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length


class PERAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.replay_buffer = PrioritizedBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.0):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        q_values = self.model.forward(state)
        action = np.argmax(q_values.cpu().detach().numpy())

        if np.random.rand() > eps:
            return self.env.action_space.sample()

        return action

    def _sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def _compute_TDerror(self, batch_size):
        transitions, idxs, weights_ = self._sample(batch_size)

        states, actions, rewards, next_states, dones = transitions

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights_ = torch.FloatTensor(weights_).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        td_errors = torch.pow(curr_Q - expected_Q, 2) * weights_

        return td_errors, idxs

    def update(self, batch_size):
        td_errors, idxs = self._compute_TDerror(batch_size)

        td_errors_mean = td_errors.mean()

        self.optimizer.zero_grad()
        td_errors_mean.backward()
        self.optimizer.step()

        for idx, td_error in zip(idxs, td_errors.cpu().detach().numpy()):
            self.replay_buffer.update_priority(idx, td_error)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        if type(state) is tuple:
            state = state[0]
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print(f"Episode {episode}: {episode_reward}")
                break

        state = next_state

    return episode_rewards


MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

env = gymnasium.make("CartPole-v1")
agent = PERAgent(env)

mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
