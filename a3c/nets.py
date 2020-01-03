from torch import nn
import torch
import math


class MyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.linear_layer = nn.Sequential(
            nn.Linear(state_space, 200),
            nn.ReLU6()
        )
        self.mu = nn.Linear(200, action_space)
        self.sigma = nn.Linear(200, action_space)
        self.value_layer = nn.Sequential(
            nn.Linear(state_space, 100),
            nn.ReLU6(),
            nn.Linear(100, 1)
        )
        self.distribution = torch.distributions.Normal
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.mseloss = nn.MSELoss()

    def forward(self, data):
        linear_layer = self.linear_layer(data)
        mu = 2 * self.tanh(self.mu(linear_layer))
        sigma = self.softplus(self.sigma(linear_layer)) + 0.001  # avoid 0
        value = self.value_layer(data)
        return mu, sigma, value

    def select_action(self, state):
        mu, sigma, _ = self.forward(state)
        m = self.distribution(mu.reshape(1, ).data, sigma.reshape(1, ).data)
        return m.sample().numpy()

    def get_loss(self, state, action, v_t):
        mu, sigma, values = self.forward(state)
        td = v_t - values
        value_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(action)

        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration;m.scale = sigma;m.loc = mu
        exp_v = log_prob * td.detach() + 0.005 * entropy
        action_loss = -exp_v
        total_loss = (action_loss + value_loss).mean()
        return total_loss


if __name__ == '__main__':
    net = MyNet(3, 1)
    params = sum([param.numel() for param in net.parameters()])
    print(params)
