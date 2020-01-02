import torch
import torch.nn as nn
from a3c.utils import  push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from a3c.shared_adam import SharedAdam
import gym
import math, os
from a3c.nets import MyNet
from torch.distributions import Categorical
import numpy as np
from a3c.config import args

# 只使用一个线程
# os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


class Trainer(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, net_path, number):
        super().__init__()
        self.net_path = net_path
        self.name = 'w%i' % number
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.local_net = MyNet(args.state_dim, args.action_dim)  # local network
        if os.path.exists(net_path):
            self.local_net.load_state_dict(torch.load(net_path))
        else:
            self.local_net.apply(self.weight_init)
        self.env = gym.make(args.env_name).unwrapped
        self.gamma = args.gamma

    def weight_init(self, net):
        if isinstance(net, nn.Linear):
            nn.init.normal_(net.weight, mean=0., std=0.1)
            nn.init.constant_(net.bias, 0.)

    def run(self):
        total_step = 1
        while self.global_ep.value < args.MAX_EP:
            state = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            epoch_r = 0.
            # 每一轮最多走多少步
            for t in range(args.MAX_EP_STEP):
                if self.name == 'w0':
                    self.env.render()
                state = torch.tensor(state, dtype=torch.float32)
                action = self.local_net.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if t == args.MAX_EP_STEP - 1:
                    done = True
                epoch_r += reward
                buffer_state.append(state)
                action = torch.tensor(action)
                buffer_action.append(action)
                # 回报的取值在[0,-16.2736044]之间，归一化到[-1,1]之间
                buffer_reward.append((reward + 8.1) / 8.1)  # normalize

                if total_step % args.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.optimizer, self.local_net, self.global_net, done, next_state, buffer_state,
                                  buffer_action, buffer_reward, self.gamma)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:  # done and print information
                        record(self.global_ep, self.global_ep_r, epoch_r, self.res_queue, self.name)
                        break
                state = next_state
                total_step += 1
            torch.save(self.global_net.state_dict(), self.net_path)
        self.res_queue.put(None)


if __name__ == '__main__':
    net_path = 'models/net.pth'
    global_net = MyNet(args.state_dim, args.action_dim)  # global network
    if os.path.exists(net_path):
        global_net.load_state_dict(torch.load(net_path))
    # global_net.share_memory()  # share the global parameters in multiprocessing
    # optimizer = SharedAdam(global_net.parameters(), lr=0.0002)  # global optimizer
    optimizer = torch.optim.Adam(global_net.parameters(), lr=0.0002)  # global optimizer
    # 在共享内存中放入全局次数以及全局价值总和
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # 构建多个进程同时训练
    workers = [Trainer(global_net, optimizer, global_ep, global_ep_r, res_queue, net_path, i) for i in
               range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    # 后面的任务要等主进程结束后才执行，子进程不用除非进程里用了lock
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
