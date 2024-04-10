import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium.envs.registration import register
import random


def rand_argmax(v):
    a = np.max(v)
    indices = np.nonzero(v == a)[0]
    return random.choice(indices)


register(
    id="FrozenLake-v3",
    entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

env = gymnasium.make("FrozenLake-v3")


Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 1000

rList = []


state = env.reset()


print(Q[state[0] :])


# for _ in range(num_episodes):
#     state = env.reset()
#     rAll = 0
#     done = False

#     while not done:

#         action = rand_argmax(Q[state, :])
#         new_state, reward, done, _, _ = env.step(action)

#         Q[state[0], action] = np.max(Q[new_state])

#         rAll += reward
#         state = new_state

#     rList.append(rAll)
