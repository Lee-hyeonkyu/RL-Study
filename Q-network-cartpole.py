import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from collections import deque


BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.01
MAX_EP = 25000
EPSILON_DECAY = 1000
GAMMA = 0.99
torch.manual_seed(4444)
np.random.seed(4444)
MIN_SIZE = 1000


class QNetWork(nn.Module):
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
        max_q_value = torch.argmax(q_values)
        action = max_q_value.detach().item()

        return action


sdsd = []

env = gymnasium.make("CartPole-v1")
episode_reward = 0.0
episode = 0
reward_buffer = deque([0.0], maxlen=100)
state = env.reset()

q_net = QNetWork(env)

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-1)

step = 0


REWARD_ACC = []
LOSS_ACC = []

while step != MAX_EP:
    if type(state) is tuple:
        state = state[0]
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    random_sample = random.random()

    if random_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = q_net.act(state)

    new_state, reward, done, _, _ = env.step(action)

    episode_reward += reward

    state = new_state

    if done:
        state = env.reset()
        if type(state) is tuple:
            state = state[0]
        reward_buffer.append(episode_reward)
        episode_reward = 0

    state_t = torch.as_tensor(state, dtype=torch.float32)
    action_t = torch.as_tensor(action, dtype=torch.int64).unsqueeze(-1)
    reward_t = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(-1)
    done_t = torch.as_tensor(done, dtype=torch.float32).unsqueeze(-1)
    new_state_t = torch.as_tensor(new_state, dtype=torch.float32)

    target_q_values = q_net.forward(new_state_t)
    max_target_q_values = target_q_values.max(dim=0, keepdim=True)[0]
    targets = reward_t + GAMMA * (1 - done_t) * max_target_q_values
    q_values = q_net.forward(state_t)
    action_q_values = torch.gather(q_values, dim=0, index=action_t)

    # print("targets:", targets)
    # print("q_values:", q_values)
    # print("actions_q_values", action_q_values)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print()
        print("Step", step)
        print(
            "Avg Reward", np.mean(reward_buffer)
        )  # maximum length of reward_buffer is 100. Therefore, np.mean(reward_buffer) averages lastest 100 rewards
        print("Loss", loss)
        REWARD_ACC.append(np.mean(reward_buffer))
        LOSS_ACC.append(loss.item())
    step += 1


print(sum(reward_buffer) / MAX_EP)
