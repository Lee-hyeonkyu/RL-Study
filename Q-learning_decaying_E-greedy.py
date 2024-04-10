import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium.envs.registration import register

register(
    id="FrozenLake-v3",
    entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

env = gymnasium.make("FrozenLake-v3")


Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
param = 100
discount_factor = 0.99
rList = []


def q_learing_with_dis():
    for i in range(num_episodes):

        state = env.reset()
        rAll = 0
        done = False

        while not done:
            if type(state) is tuple:
                state = state[0]

            action = np.argmax(
                Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1)
            )

            new_state, reward, done, _, _ = env.step(action)

            Q[state, action] = reward + discount_factor * np.max(Q[new_state, :])

            rAll += reward
            state = new_state

        rList.append(rAll)
    return rList


# e-greedy
def q_learing_with_eps():
    for i in range(num_episodes):

        epsilon = 1.0 / ((i // param) + 1)

        state = env.reset()
        rAll = 0
        done = False

        while not done:
            if type(state) is tuple:
                state = state[0]
            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, done, _, _ = env.step(action)

            Q[state, action] = reward + discount_factor * np.max(Q[new_state, :])

            rAll += reward
            state = new_state

        rList.append(rAll)
    return rList


rList = q_learing_with_dis()
# rList = q_learing_with_eps()

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
