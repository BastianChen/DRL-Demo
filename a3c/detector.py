import torch
from a3c.nets import MyNet
from a3c.config import args
import gym


class Detector:
    def __init__(self, actor_path, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MyNet(state_dim, action_dim)
        self.net.load_state_dict(torch.load(actor_path))

    def detect(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.net.select_action(state)
        return action


if __name__ == '__main__':
    env = gym.make(args.env_name).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    detector = Detector('models/net.pth', state_dim, action_dim)
    state = env.reset()
    while True:
        action = detector.detect(state)
        next_state, reward, done, info = env.step(action)
        env.render()
        state = next_state
