import gym, torch
import numpy as np
from itertools import count
from torch import nn, optim
from torch.distributions import Categorical


class ACModel1:
    def __init__(self, env, render=True):
        # 定义环境
        self.env = env
        # 定义状态形状(4)
        self.state_space = env.observation_space.shape[0]
        # 定义动作个数(2)
        self.action_space = env.action_space.n
        # 定义折扣率
        self.gamma = 0.99
        self.render = render
        # 定义一个最小正数eps用于分母相加，防止精度丢失的问题
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 10
        self.net = MyNet(self.state_space, self.action_space)
        self.optimizer = optim.Adam(self.net.parameters(), lr=3e-2)
        self.mseloss = nn.MSELoss()

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob, action_value = self.net(state)
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(action_prob)
        # 从分布中随机取样
        action = m.sample()
        self.net.actions.append([m.log_prob(action), action_value])
        return action.item()

    def get_v_value(self):
        R = 0
        v_values = []

        for reward in self.net.rewards[::-1]:
            R = reward + self.gamma * R
            v_values.insert(0, R)

        v_values = torch.tensor(v_values)
        # 根据期望和方差做标准归一化
        v_value = (v_values - v_values.mean()) / (v_values.std() + self.eps)
        return v_value

    def train(self):
        # 从1开始无限循环
        for i in count(1):
            # 重置网络
            state = self.env.reset()
            epoch_reward = 0
            # 每一个批次最多运行9999次
            for j in range(1, 10000):
                # 选择下一个动作
                action = self.select_action(state)
                # 执行动作,获取环境状态、奖励、终止信号和当前生命值等信息
                state, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                self.net.rewards.append(reward)
                epoch_reward += reward
                if done:
                    break
            # 更新回报
            self.running_reward = 0.05 * epoch_reward + (1 - 0.05) * self.running_reward
            v_value = self.get_v_value()
            policy_losses = []  # list to save actor (policy) loss
            value_losses = []  # list to save critic (value) loss
            for (log_prob, value), R in zip(self.net.actions, v_value):
                # 求得动作优势
                advantage = R - value.item()
                # 用动作优势给决策打分
                policy_losses.append(-log_prob * advantage)
                value_losses.append(self.mseloss(value, R))

            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.net.actions = []
            self.net.rewards = []

            if i % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i, epoch_reward, self.running_reward))

            # check if we have "solved" the cart pole problem
            if self.running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(self.running_reward, j))
                break


class MyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU()
        )
        self.action = nn.Sequential(
            nn.Linear(128, action_space),
            nn.Softmax(-1)
        )
        self.value = nn.Linear(128, 1)
        self.actions = []
        self.rewards = []

    def forward(self, data):
        data = self.linear_layer(data)
        # 预测的动作
        action_prob = self.action(data)
        # 预测动作的评分
        action_value = self.value(data)
        return action_prob, action_value


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped  # 解除动作限制
    net = ACModel1(env)
    net.train()
