import gym
from pong.nets import MainNet
from torchvision import transforms
import torch
from tensorboardX import SummaryWriter
from torch import nn
import cv2
import numpy as np
from PIL import Image
import os
from collections import deque
import random


class Trainer:
    def __init__(self, env, net_path, is_render=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_render = is_render
        self.current_net = MainNet().to(self.device)
        self.target_net = MainNet().to(self.device)
        self.env = env
        self.epochs = 4000000
        self.buffer_size = 6000
        self.start_epsilon = 0.1
        self.end_spsilon = 0.0001
        self.batch_size = 128
        self.gamma = 0.99
        self.observe = 1000
        self.trans = transforms.ToTensor()
        self.net_path = net_path
        self.optimizer = torch.optim.Adam(self.current_net.parameters(), weight_decay=0.0005)
        self.buffer = deque(maxlen=self.buffer_size)
        self.writer = SummaryWriter()
        if os.path.exists(net_path):
            self.current_net.load_state_dict(torch.load(net_path))
            self.target_net.load_state_dict(torch.load(net_path))
        else:
            self.current_net.apply(self.init_weight)
            self.target_net.apply(self.init_weight)

    def init_weight(self, net):
        if isinstance(net, nn.Linear) or isinstance(net, nn.Conv2d):
            nn.init.normal_(net.weight, mean=0, std=0.1)
            nn.init.constant_(net.bias, 0.)

    def edit_image(self, image, width=84, height=84):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image[34:194, :], (width, height))
        pimage = Image.fromarray(image)
        pimage = self.trans(pimage)
        return pimage

    def train(self):
        # 初始化状态
        state = self.env.reset()
        state = self.edit_image(state)
        state = torch.stack([state for _ in range(4)], dim=1).to(self.device)
        for i in range(self.epochs):
            if self.is_render:
                self.env.render()
            if i <= self.observe:
                action = np.random.randint(0, 6)
            else:
                epsilon = self.end_spsilon + ((self.epochs - i) * (self.start_epsilon - self.end_spsilon) / self.epochs)
                if np.random.random() <= epsilon:
                    action = np.random.randint(0, 6)
                else:
                    action = torch.argmax(self.current_net(state))
            next_state, reward, done, _ = self.env.step(action)
            if done:
                state_new = self.env.reset()
                state_new = self.edit_image(state_new)
                next_state = torch.cat((state[0, 1:, :, :], state_new)).unsqueeze(0)
            else:
                next_state = self.edit_image(next_state)
                next_state = torch.cat((state[0, 1:, :, :], next_state)).unsqueeze(0)
            self.buffer.append([state, action, reward, next_state, done])
            state = next_state

            # 训练代码
            if i > self.observe:
                data_batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*data_batch)
                state_batch = torch.cat(state_batch).to(self.device)
                action_batch = torch.tensor(action_batch).to(self.device)
                reward_batch = torch.Tensor(reward_batch).to(self.device)
                next_state_batch = torch.cat(next_state_batch).to(self.device)
                done_batch = torch.Tensor(done_batch).to(self.device)

                if i % 30 == 0:
                    self.target_net.load_state_dict(self.current_net.state_dict())

                current_prediction = torch.argmax(self.current_net(next_state_batch), dim=-1)
                current_q = self.current_net(state_batch)
                current_q = current_q.gather(1, action_batch.unsqueeze(1))
                target_q = self.target_net(next_state_batch).gather(1, current_prediction.unsqueeze(1)).squeeze(1)
                target_q = torch.stack([reward if done else reward + self.gamma * q for reward, q, done in
                                        zip(reward_batch, target_q, done_batch)])
                loss = self.current_net.get_loss(self.writer, i, current_q, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"epoch:{i},loss:{loss.item()}")

                # 保存网络权重
                if (i - 1) % 100 == 0:
                    torch.save(self.current_net.state_dict(), self.net_path)


if __name__ == '__main__':
    env = gym.make("Pong-v0").unwrapped
    trainer = Trainer(env, "models/net.pth")
    trainer.train()
