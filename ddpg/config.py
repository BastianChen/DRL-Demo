import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="Pendulum-v0", help="环境名")
parser.add_argument('--tau', default=0.005, type=float, help="用于target网络更新的平滑系数")
parser.add_argument('--gamma', default=0.99, type=float, help="折扣率")
parser.add_argument('--buffer_size', default=20000, type=int, help="样本池大小")
parser.add_argument('--render', default=True, type=bool, help="是否显示环境界面")
parser.add_argument('--exploration_noise', default=0.1, type=float,
                    help='探索噪声，在网络还没有开始训练前，将图片输入网络输出来的值都差不多，加上这个噪声可以使动作值更多样化，使得样本池中的数据联系性更少')
parser.add_argument('--max_length_of_trajectory', default=3000, type=int, help="每一轮的最大步数")
parser.add_argument('--batch_size', default=512, type=int, help="从样本池中取多少样本")
parser.add_argument('--update_iteration', default=10, type=int)
# parser.add_argument('--observe', default=4000, type=int, help="刚运行时随机采样的轮次")
args = parser.parse_args()
