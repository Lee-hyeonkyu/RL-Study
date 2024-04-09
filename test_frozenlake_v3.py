
import gym
from gym.envs.registration import register
import readchar 
import colorama as cr


cr.init(autoreset=True)
gym.envs.registration.register(id='FrozenLake-v3',
entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4', 'is_slippery':False})

env = gym.make('FrozenLake-v3', render_mode='ansi')
env.reset()
print(env.render())

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}


while True:
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
    action = arrow_keys[key]
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.render())
    print(f"State: {observation} Action: {action} Reward: {reward} Info: {info} ")
    
    if terminated:
        print("Finished with reward", reward)
        break
    