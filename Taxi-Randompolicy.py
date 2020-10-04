import sys

import gym as gym
import random
import numpy as np

import gym as gym
from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v3").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])
env.reset() # reset environment to a new, random state

all_epochs = []
all_penalties = []

epochs = 0

penalties, reward = 0, 0
total_epochs, total_penalties, total_rewards = 0, 0, 0
episodes = 1000

frames = []  # for animation

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

for i in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1


        total_rewards += reward



        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
        )
    #print_frames(frames)

    total_penalties += penalties
    total_epochs += epochs


print(f"Results after {episodes} episodes:")
print(f"Average rewards per episode: {total_rewards / episodes}")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))