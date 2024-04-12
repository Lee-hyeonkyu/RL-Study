import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium.envs.registration import register

register(
    id="FrozenLake-v3",
    entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},  # change
)

env = gymnasium.make("FrozenLake-v3")


Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
discount_factor = 0.99
learning_rate = 0.85
rList = []


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

        # $Q(s,a) \leftarrow(1-\alpha)Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a')]$
        Q[state, action] = (1 - learning_rate) * np.max(
            Q[state, action]
        ) + learning_rate * (reward + discount_factor * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    rList.append(rAll)


print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
