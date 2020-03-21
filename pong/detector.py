import gym
from pong.nets import MainNet
from torchvision import transforms
import torch
import cv2
from PIL import Image


class Detector:
    def __init__(self, env, net_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 如果没有GPU的话把在GPU上训练的参数放在CPU上运行，cpu-->gpu 1:lambda storage, loc: storage.cuda(1)
        self.map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
        self.net = MainNet().to(self.device)
        self.net.load_state_dict(torch.load(net_path, map_location=self.map_location))
        self.net.eval()
        self.env = env
        self.trans = transforms.ToTensor()

    def edit_image(self, image, width=84, height=84):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image[34:194, :], (width, height))
        pimage = Image.fromarray(image)
        pimage = self.trans(pimage)
        return pimage

    def detect(self):
        # 初始化状态
        state = self.env.reset()
        state = self.edit_image(state)
        state = torch.stack([state for _ in range(4)], dim=1).to(self.device)
        while True:
            try:
                self.env.render()
                prediction = self.net(state)
                action = torch.argmax(prediction).item()
                print(action)

                next_state, reward, done, _ = self.env.step(action)

                if done:
                    state_new = self.env.reset()
                    state_new = self.edit_image(state_new)
                    next_state = torch.cat((state[0, 1:, :, :], state_new)).unsqueeze(0)
                else:
                    next_state = self.edit_image(next_state)
                    next_state = torch.cat((state[0, 1:, :, :], next_state)).unsqueeze(0)
                state = next_state
                # if reward == 1:
                #     checkpoint += 1
                # if terminal:
                #     print(f"飞行到第{checkpoint}关")
                #     checkpoint = 0
            except KeyboardInterrupt:
                print("Quit")


if __name__ == '__main__':
    env = gym.make("Pong-v0").unwrapped
    detector = Detector(env, "models/net_3000.pth")
    detector.detect()
