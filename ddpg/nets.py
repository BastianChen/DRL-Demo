from torch import nn
import torch
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(state_dim, 1200),
            nn.LeakyReLU(),
            nn.Linear(1200, 600),
            nn.LeakyReLU(),
            nn.Linear(600, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, data):
        # 将值域从[-1,1]变为[-2,2]
        data = self.linear(data)
        data = self.max_action * data
        return data


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(state_dim + action_dim, 1200),
            nn.LeakyReLU(),
            nn.Linear(1200, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 1)
        )

    def forward(self, state, action):
        data = torch.cat((state, action), dim=-1)
        data = self.linear(data)
        return data


if __name__ == '__main__':
    a = torch.Tensor(2, 784)
    print(a.shape)
    b = nn.Linear(28 * 28, 1200)
    c = b(a)
    print(c.shape)
    d = nn.BatchNorm1d(1200)
    c = d(c)
    print(c.shape)
