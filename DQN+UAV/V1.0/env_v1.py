from tqdm import tqdm
import torch
import collections
import random
import math
import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        # 定义状态空间和动作空间
        #状态空间x,y,v,航向角
        self.d=50   #测距仪的范围
        self.observation_space = gym.spaces.Box(low=np.array([ -1*np.pi]), high=np.array([np.pi]), dtype=np.float32)
        #地图边界
        self.space1 = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1000, 1000]), dtype=np.float32)
        #动作空间角速度
        self.action_space = gym.spaces.Discrete(3)
        self.change = [-0.2*np.pi, 0, 0.2*np.pi]#转向信号

        #UAV的绝对坐标
        self.xy=[100,100]
        # 定义起点、终点和障碍物
        self.heading = 0.5*np.pi*0.5
        self.goal = [900, 900]
        #匀速
        self.v = 20
        self.t_step = 1  # 时间步


        #状态空间
        self.drone_state = [np.pi]
        self.normalized = [2*np.pi, 1200, np.pi]
        self.observation = [np.pi]

        # 用于存储环境信息的字典
        self.info = {"collision": False}

        # 定义奖励函数中的常数
        self.goal_reward = 1000
        self.step_penalty = -6
        self.r_goal = 10 #目标半径S
        self.done1 = False
    def step(self, action):
        # 保存上一步的状态
        prev_state = np.array(self.drone_state)
        d_lod = self.get_d2goal()
        #print(type(self.drone_state[0]))
        # 更新状态
        self.heading += self.change[action]# 更新航向角
        self.xy[0] += self.v * self.t_step * math.cos(self.heading) # 更新x坐标
        self.xy[1] += self.v * self.t_step * math.sin(self.heading)  # 更新y坐标
        self.drone_state[0] = self.get_angle2goal()

        # 判断是否到达终点
        self.determine()
        # 计算奖励
        reward = self.calculate_reward(d_lod)
        #标准化状态空间
        for i in range(1):
            self.observation[i] = self.drone_state[i]
            #self.observation[i] = (self.drone_state[i] - 0.5 * self.normalized[i]) /(0.5 * self.normalized[i])
            #self.observation[i] = self.drone_state[i] / self.normalized[i]
        return self.observation, reward, self.done1, self.info


    def reset(self):
        # 重置环境，返回初始状态
        self.done1 = False
        # 初始化无人机状态
        self.xy = [100, 100]
        self.heading = 0
        self.drone_state[0] = self.get_angle2goal()
        # 标准化状态空间
        for i in range(1):
            self.observation[i] = self.drone_state[i]
            #self.observation[i] = (self.drone_state[i] - 0.5 * self.normalized[i]) / (0.5 * self.normalized[i])
            #.observation[i] = self.drone_state[i] / self.normalized[i]

        return self.observation

    def render(self, mode='human'):
        if mode == 'human':
            # Human-readable rendering (e.g., visual display)
            self._human_render()
        elif mode == 'rgb_array':
            # Return an RGB array for rendering in automated testing
            return self._rgb_array_render()
        else:
            super().render(mode=mode)

    def calculate_reward(self,d_lod):# 计算奖励

            d_new = self.get_d2goal()
            return (d_lod - d_new)*2 + self.step_penalty
    def get_angle(self):#返回航向角
        if self.heading > 2*np.pi:
            self.heading -= 2*np.pi

        return self.heading
    def get_d8(self, k):#返回8个测距仪的数据
        return 50
    def get_d2goal(self):#返回到目标点距离
        dx = self.xy[0] - self.goal[0]
        dy = self.xy[1] - self.goal[1]
        # 使用勾股定理计算距离
        distance = math.sqrt(pow(dx, 2) + pow(dy, 2))
        return distance
    def get_angle2goal(self):#返回到目标点的角度
        dx = self.xy[0] - self.goal[0]
        dy = self.xy[1] - self.goal[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        diff = (self.heading - angle_rad+ math.pi) % (2 * math.pi) - math.pi
        return diff
    def determine(self):

        if (self.get_d2goal() < self.r_goal):
            self.done1 = True
         #判断是否超出边界
        if (self.xy[0] < self.space1.low[0] or self.xy[0] > self.space1.high[0] or self.xy[1] < self.space1.low[1] or self.xy[1] > self.space1.high[1]):
            #print("222")
            self.done1 = True

    def _human_render(self):
        # Implement human-readable rendering logic here
        plt.scatter(self.xy[0], self.xy[1], marker='o', color='red', label='Drone')
        #plt.show()

    def _rgb_array_render(self):
        # Implement rendering logic to return an RGB array
        # This can be used for automated testing
        # (Similar to _human_render, but instead of displaying, return the RGB array)
        pass