import argparse
import gym

env = gym.make('Pendulum-v0')
# 定义状态和动作的数量
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='Pendulum-v0')
parser.add_argument('--gamma', default=0.9, type=float, help="折扣率")
parser.add_argument('--state_dim', default=state_dim, type=int)
parser.add_argument('--action_dim', default=action_dim, type=int)
parser.add_argument('--max_action', default=max_action, type=int)
parser.add_argument('--UPDATE_GLOBAL_ITER', default=5, type=int, help='全局参数隔几轮更新一次')
parser.add_argument('--MAX_EP', default=30000, type=int, help='局部最大轮次')
parser.add_argument('--MAX_EP_STEP', default=200, type=int, help='每一轮次最多执行步数')
args = parser.parse_args()
