from itertools import count
import os
import numpy as np
import gym
from torch import nn
import torch
from ddpg.nets import Actor, Critic
from collections import deque
from ddpg.config import args
from tensorboardX import SummaryWriter
import random


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=args.buffer_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Trainer:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        if os.path.exists('models/actor1.pth'):
            self.actor.load_state_dict(torch.load('models/actor1.pth'))
            self.critic.load_state_dict(torch.load('models/critic1.pth'))

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.mseloss = nn.MSELoss()
        self.action_dim = action_dim
        self.replay_memory = deque(maxlen=args.buffer_size)
        self.writer = SummaryWriter()

    def weight_init(self, net):
        if isinstance(net, nn.Linear):
            nn.init.normal_(net.weight)
            nn.init.constant_(net.bias, 0)

    def select_action(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().detach().numpy()  # 将矩阵组装成一维的

    def train(self):
        ep_reward = 0
        loss_update_index = 1
        for i in range(100000):
            state = env.reset()
            # 制造样本池
            for t in count():
                action = self.select_action(state)
                # 使用np.clip()进行裁剪，使得小于low的数等于low，大于high的数等于high。
                action = (action + np.random.normal(0, 0.1, size=self.action_dim)).clip(env.action_space.low,
                                                                                        env.action_space.high)
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                env.render()
                self.replay_memory.append([state, next_state, action, reward, np.float(done)])
                # self.replay_memory.push((state, next_state, action, reward, np.float(done)))
                state = next_state
                if done or t >= args.max_length_of_trajectory:
                    self.writer.add_scalar('ep_reward', ep_reward / t, global_step=i)
                    print("Ep_i \t{}, the ep_reward is \t{:0.2f}, the step is \t{}".format(i, ep_reward / t, t))
                    ep_reward = 0
                    break
            if len(self.replay_memory) >= args.buffer_size - 1:
                for j in range(args.update_iteration):
                    batch_data = random.sample(self.replay_memory, min(len(self.replay_memory), args.batch_size))
                    state_batch, next_state_batch, action_batch, reward_batch, done_batch = zip(*batch_data)
                    state_batch, next_state_batch, action_batch, reward_batch, done_batch = torch.FloatTensor(
                        state_batch).to(self.device), torch.FloatTensor(next_state_batch).to(
                        self.device), torch.FloatTensor(action_batch).to(self.device), torch.FloatTensor(
                        reward_batch).to(self.device), torch.FloatTensor(done_batch).to(self.device)

                    # 计算目标网络的估计Q值
                    target_Q = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
                    # 计算实际Q值
                    target_Q = reward_batch.reshape((-1, 1)) + (
                            (1 - done_batch.reshape((-1, 1))) * args.gamma * target_Q)
                    # target_Q = reward_batch + ((1 - done_batch) * args.gamma * target_Q)

                    # 计算当前网络的估计Q值
                    current_Q = self.critic(state_batch, action_batch)
                    critic_loss = self.mseloss(current_Q, target_Q)
                    self.writer.add_scalar('loss/critic_loss', critic_loss, loss_update_index)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
                    self.writer.add_scalar('loss/actor_loss', actor_loss, loss_update_index)
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # 更新target网络的权重
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    loss_update_index += 1
                    if i % 10 == 0:
                        torch.save(self.actor.state_dict(), 'models/actor1.pth')
                        torch.save(self.critic.state_dict(), 'models/critic1.pth')


if __name__ == '__main__':
    env = gym.make(args.env_name).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    train = Trainer(state_dim, action_dim, max_action)
    train.train()
