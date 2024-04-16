import gymnasium

env = gymnasium.make("CartPole-v1", render_mode="human")
env.reset()
random_episodes = 0
reward_sum = 0

while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(observation, reward, done)
    reward_sum += reward

    if done:
        random_episodes += 1
        print(f"Reward for this episodde was: {reward_sum}")
        reward_sum = 0
        env.reset()
