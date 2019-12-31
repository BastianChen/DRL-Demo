import gym
from ddpg.nets import Actor
from itertools import count
from ddpg.config import args
import torch


class Detector:
    def __init__(self, actor_path, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor.load_state_dict(torch.load(actor_path))

    def detect(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().detach().numpy()


if __name__ == '__main__':
    env = gym.make(args.env_name).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    # detector = Detector('models1/actor.pth', state_dim, action_dim, max_action)
    detector = Detector('models1/actor3.pth', state_dim, action_dim, max_action)
    for i in range(10000):
        state = env.reset()
        ep_reward = 0
        for t in count():
            action = detector.detect(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            state = next_state
            ep_reward += reward
            if done or t >= args.max_length_of_trajectory:
                print("epoch:{},ep_reward:{}".format(i, ep_reward / t))
                break
