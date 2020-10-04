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


alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

epochs = 0

penalties, reward = 0, 0
total_epochs, total_penalties, total_rewards = 0, 0, 0
episodes = 23

for i in range(1, 400):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        total_rewards += reward
        epochs += 1

    total_penalties += penalties

    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average rewards per episode: {total_rewards / episodes}")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))