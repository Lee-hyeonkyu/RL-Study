import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from collections import deque

BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
MAX_EP = 25000
discount_factor = 0.99
torch.manual_seed(1234)
np.random.seed(1234)
MIN_SIZE = 1000


class QNetWork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # input_size = env.observation_space.n
        input_size = int(np.prod(env.observation_space.shape))
        output_size = env.action_space.n
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.Tanh(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self.forward(state_t.unsqueeze(0))  # [x,x,x,x]
        max_q_idx = torch.argmax(q_values, dim=-1)

        action = max_q_idx.detach().item()  # 1,2,3,4
        return action


env = gymnasium.make("FrozenLake-v1")
q_net = QNetWork(env)
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-1)
state = env.reset()
Q = np.zeros([env.observation_space.n, env.action_space.n])

rList = []
episode = 0
step = 0
rAll = 0.0

re_data = deque()
for _ in range(MIN_SIZE):
    if type(state) is tuple:
        state = state[0]
    action = env.action_space.sample()
    new_state, reward, done, _, _ = env.step(action)
    transition = (state, action, reward, done, new_state)
    re_data.append(transition)
    state = new_state

    if done:
        state = env.reset()

state = env.reset()

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
    transition = (state, action, reward, done, new_state)
    re_data.append(transition)
    Q[state, action] = reward + discount_factor * np.max(Q[new_state, :])

    rAll += reward
    state = new_state

    if done:
        state = env.reset()
        if type(state) is tuple:
            state = state[0]
        rList.append(rAll)
        rAll = 0

    transitions = random.sample(re_data, BATCH_SIZE)

    states = np.asarray([t[0] for t in transitions])
    action = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_states = np.asarray([t[4] for t in transitions])

    state_t = torch.as_tensor(state, dtype=torch.float32)
    action_t = torch.as_tensor(action, dtype=torch.int64)
    reward_t = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(-1)
    done_t = torch.as_tensor(done, dtype=torch.float32).unsqueeze(-1)
    new_state_t = torch.as_tensor(new_state, dtype=torch.float32)

    target_q_values = q_net.forward(new_state_t.unsqueeze(0))
    max_target_q_values = target_q_values.max(dim=0, keepdim=True)[0]
    targets = reward_t + discount_factor * (1 - done_t) * max_target_q_values

    q_values = q_net.forward(state_t.unsqueeze(0))

    action_q_values = torch.gather(input=q_values, dim=0, index=action_t)

    loss = nn.functional.mse_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print()
        print("Step", step)
        print("Avg Reward", np.mean(rList))
        print("Loss", loss)

    step += 1


print("Success rate: " + str(sum(rList) / MAX_EP))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
