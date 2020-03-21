import gym, torch
from torch import nn
from itertools import count
from torch.distributions import Categorical
import numpy as np


class ACModel2:
    def __init__(self, env, render=True):
        self.env = env
        self.render = render
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.net = MyNet(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.mseloss = nn.MSELoss()

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action, value = self.net(state)
        c = Categorical(action)
        action = c.sample()
        log_prob = c.log_prob(action)
        self.net.actions.append([log_prob, value])
        return action.item()

    def get_v_value(self):
        R = 0
        v_values = []

        for reward in self.net.rewards[::-1]:
            R = reward + self.gamma * R
            v_values.insert(0, R)

        v_values = torch.tensor(v_values)
        v_values = (v_values - v_values.mean()) / (v_values.std() + self.eps)
        return v_values

    def train(self):
        for i in count(1):
            state = self.env.reset()
            epoch_reward = 0
            for j in count(1):
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                self.net.rewards.append(reward)
                epoch_reward += reward
                if done:
                    print(f"epoch:{i},run step is {j}")
                    break
                # if j % 10 == 0:
                #     print(f'step:{j},reward:{reward}')
            v_value = self.get_v_value()
            policy_losses = []
            critic_losses = []
            for (log_prob, value), reward in zip(self.net.actions, v_value):
                advantage = reward - value.item()
                policy_losses.append(-log_prob * advantage)
                critic_losses.append(self.mseloss(value, reward))

            loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.net.actions = []
            self.net.rewards = []


class MyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU()
        )
        self.action_layer = nn.Sequential(
            nn.Linear(128, action_space),
            nn.Softmax(-1)
        )
        self.value_layer = nn.Linear(128, 1)
        self.actions = []
        self.rewards = []

    def forward(self, data):
        data = self.linear_layer(data)
        action = self.action_layer(data)
        value = self.value_layer(data)
        return action, value


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    net = ACModel2(env)
    net.train()
