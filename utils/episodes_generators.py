import gymnasium as gym
import random


class MountainCarEpisodes:
    def __init__(self, num_episodes=100):
        self.num_episodes = num_episodes
        self.env = gym.make('MountainCar-v0')
        self.episode_data = []

    def run_episodes(self):
        for i_episode in range(self.num_episodes):
            observation = self.env.reset()
            observation = observation[0]
            episode_reward = 0
            steps = 0
            episode_states = []
            while True:
                # Choose action based on position and velocity
                if observation[0] < -0.5:
                    action = 2  # push right
                elif observation[0] > 0.5:
                    action = 0  # push left
                else:
                    action = 2 if observation[1] > 0 else 0  # push in the direction of velocity

                observation, reward, done, info, _ = self.env.step(action)
                episode_states.append((observation, reward, steps))
                episode_reward += reward
                steps += 1
                if done or steps > 199:
                    # print(f"Episode finished after {steps} timesteps with reward {episode_reward}")
                    self.episode_data.append(episode_states)
                    break
        print(f"Episode generation done!")

    def get_episode_data(self):
        return self.episode_data


class CartPoleEpisodes:
    def __init__(self, num_episodes=100):
        self.num_episodes = num_episodes
        self.env = gym.make('CartPole-v1')
        self.episode_data = []

    def run_episodes(self):
        for i_episode in range(self.num_episodes):
            observation = self.env.reset()
            observation = observation[0]
            episode_reward = 0
            steps = 0
            episode_states = []
            while True:
                # Choose action based on pole angle
                if random.random() > 0.0:
                    action = 0 if observation[2] + observation[3] < 0 else 1
                else:
                    action = int(random.random() > 0.5)
                observation, reward, done, info, _ = self.env.step(action)
                episode_states.append((observation, action, reward, steps))
                episode_reward += reward
                steps += 1
                if done or episode_reward > 500:
                    # print(f"Episode finished after {steps} timesteps with reward {episode_reward}")
                    self.episode_data.append(episode_states)
                    break
        print(f"Episode generation finished")

    def get_episode_data(self):
        return self.episode_data

# mountain_car = CartPoleEpisodes(num_episodes=10000)
# mountain_car.run_episodes()
# episode_data = mountain_car.get_episode_data()
# print('123')
