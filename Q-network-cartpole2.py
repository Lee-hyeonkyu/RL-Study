import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from collections import deque


GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
MAX_EP = 25000

REWARD_ACC = list()
LOSS_ACC = list()

torch.manual_seed(4444)
np.random.seed(4444)


class QNetword(nn.Module):
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
        max_q_values = torch.argmax(q_values)
        action = max_q_values.detach().item()
        return action


env = gymnasium.make("CartPole-v1")

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

online_net = QNetword(env)
target_net = QNetword(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

state = env.reset()


for _ in range(MIN_REPLAY_SIZE):
    if type(state) is tuple:
        state = state[0]

    action = env.action_space.sample()
    new_state, reward, done, _, _ = env.step(action)
    transition = (state, action, reward, done, new_state)
    replay_buffer.append(transition)
    state = new_state

    if done:
        state = env.reset()


# Main

state = env.reset()
step = 0

while step != MAX_EP:
    if type(state) is tuple:
        state = state[0]
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    random_sample = random.random()

    if random_sample > epsilon:
        action = online_net.act(state)
    else:
        action = env.action_space.sample()

    new_state, reward, done, _, _ = env.step(action)
    transition = (state, action, reward, done, new_state)
    replay_buffer.append(transition)
    state = new_state
    episode_reward += reward
    if done:
        state = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    transitions = random.sample(replay_buffer, BATCH_SIZE)  # batch_size = 32

    states = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_states = np.asarray([t[4] for t in transitions])

    states_t = torch.as_tensor(states, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_states_t = torch.as_tensor(new_states, dtype=torch.float32)

    online_with_new_states = online_net.forward(new_states_t)
    argmax_online_with_new_states = online_with_new_states.argmax(dim=1, keepdim=True)

    offline_with_new_states = target_net.forward(new_states_t)
    target_q_values = torch.gather(
        input=offline_with_new_states, dim=1, index=argmax_online_with_new_states
    )
    targets = rewards_t + GAMMA * (1 - dones_t) * target_q_values

    q_values = online_net.forward(states_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if not step % TARGET_UPDATE_FREQ:
        target_net.load_state_dict(online_net.state_dict())

    if not step % 1000:
        print()
        print(f"Step: {step}")
        print(f"Avg Reward {np.mean(reward_buffer)}")
        print(f"Loss {loss}")
        REWARD_ACC.append(np.mean(reward_buffer))
        LOSS_ACC.append(loss.item())

    step += 1


print(sum(reward_buffer) / MAX_EP)
