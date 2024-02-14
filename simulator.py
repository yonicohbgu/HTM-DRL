import os
import random
import shutil
import string
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym

from model_src.model import Model
from utils.hyperparameters_configuration import HyperparametersConfiguration
from utils.episodes_generators import CartPoleEpisodes
from statistics import *

def handler():
    raise Exception("end of time")


rng = np.random.default_rng(2021)
np.random.seed(0)


def simulate(train_size):
    cartpoleEpisodesGenerator = CartPoleEpisodes(num_episodes=train_size)
    cartpoleEpisodesGenerator.run_episodes()
    cartpoleEpisodes = cartpoleEpisodesGenerator.get_episode_data()
    model = Model()
    now = time.time()
    for x in range(1):
        # model.learn(cartpoleEpisodes)
        model.train_net(cartpoleEpisodes)
    print(f'Train time: {time.time() - now}')

    env = gym.make('CartPole-v1')
    all_trg_len = []
    for x in range(100):
        # print(f'********{x}********')
        observation = env.reset()
        cart_pose, cart_vel, pole_ang, pole_ang_vel = observation[0]
        episode_reward = 0
        steps = 0
        while True:
            # action = model.predict(cart_pose, cart_vel, pole_ang, pole_ang_vel, steps, 1)
            action = model.predict_net(cart_pose, cart_vel, pole_ang, pole_ang_vel, steps, 1)
            optimal = 0 if pole_ang + pole_ang_vel < 0 else 1
            observation, reward, done, info, _ = env.step(action)
            episode_reward += reward
            steps += 1
            cart_pose, cart_vel, pole_ang, pole_ang_vel = observation
            if done or episode_reward > 500:
                all_trg_len.append(episode_reward)
                # print(f"Episode finished after {steps} timesteps with reward {episode_reward}")
                break
    print(f'The average length of trg is: {sum(all_trg_len)/len(all_trg_len)}')
    print(f'The std of trg length is: {stdev(all_trg_len)}')
    print(sorted(all_trg_len))

    # print("Spatial Pooler Mini-Columns")
    # print(str(model.spatial_pooler))
    # print("")
    # print("Temporal Memory Cells")
    # print(str(model.temporal_memory))
    # print("")



def delete_folder_content(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def create_folder_if_missing(path):
    isexist = os.path.exists(path)
    if not isexist:
        os.makedirs(path)


if __name__ == '__main__':
    simulate(25)
