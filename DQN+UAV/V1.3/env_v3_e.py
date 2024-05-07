from tqdm import tqdm
import torch
import collections
import random
import math
import gymnasium as gym
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DDPG
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DroneEnv(gym.Env):
    def __init__(self):
        #加载追击者模型
        self.model_p = DDPG.load("TrainedModel/Actor1.1_reaction.pkl")
        self.t=0
        super(DroneEnv, self).__init__()
        # 定义状态空间和动作空间
        #状态空间x,y,v,航向角
        #self.d=50   #测距仪的范围
        self.observation_space = gym.spaces.Box(low=np.array([ -1*np.pi]), high=np.array([np.pi]), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([ -np.pi*0.03]), high=np.array([np.pi*0.03]), dtype=np.float32)
        #地图边界
        self.space1 = gym.spaces.Box(low=np.array([0, 0]), high=np.array([2000, 2000]), dtype=np.float32)
        #动作空间角速度
        #self.action_space = gym.spaces.Discrete(3)
        #self.change = [-0.2*np.pi, 0, 0.2*np.pi]#转向信号

        #UAV的坐标
        self.xy_p=[100,100]
        self.xy_e = [900, 900]
        # 定义初始航向角
        self.heading_p = -np.pi
        self.heading_e = 0
        

        #匀速
        self.v_p = 20
        self.v_e = 10

        self.t_step = 1  # 时间步

        #状态空间，夹角
        self.drone_state_p = [np.pi]
        self.drone_state_e = [np.pi]
        #标准化用的两个数组
        # self.normalized = [2*np.pi, 1200, np.pi]
        self.observation_p = np.array([np.pi],dtype=np.float32)
        self.observation_e = np.array([np.pi], dtype=np.float32)

        # 用于存储环境信息的字典
        self.info = {"collision": False}

        # 定义奖励函数中的常数
        self.xy_e_reward = 1000
        self.step_penalty = 36
        #self.step_penalty = 1
        self.r_goal = 50 #目标半径S
        self.done1 = False
    def step(self, action):
        self.t += 1
        d_lod = self.get_d2goal()
        action_p, _states = self.model_p.predict(self.observation_p, deterministic=True)
        # 执行P动作
        self.heading_p += action_p  # 更新航向角
        self.xy_p[0] += self.v_p * self.t_step * math.cos(self.heading_p)  # 更新x坐标
        self.xy_p[1] += self.v_p * self.t_step * math.sin(self.heading_p)  # 更新y坐标
        # 执行E动作
        self.heading_e += action  # 更新航向角
        self.xy_e[0] += self.v_e * self.t_step * math.cos(self.heading_e)  # 更新x坐标
        self.xy_e[1] += self.v_e * self.t_step * math.sin(self.heading_e)  # 更新y坐标
        # 保存上一步的状态
        prev_state = np.array(self.drone_state_p)
        #print(type(self.drone_state_p[0]))
        # 更新状态
        self.drone_state_p[0] = self.get_angle2goal()
        self.drone_state_e[0] = self.get_angle2p()

        # 判断是否到达终点
        self.determine()
        # 计算奖励
        reward = self.calculate_reward(d_lod)
        #标准化状态空间
        for i in range(1):
            self.observation_e[i] = self.drone_state_e[i]
            self.observation_p[i] = self.drone_state_p[i]
            #self.observation[i] = (self.drone_state_p[i] - 0.5 * self.normalized[i]) /(0.5 * self.normalized[i])
            #self.observation[i] = self.drone_state_p[i] / self.normalized[i]
        return self.observation_e, reward, self.done1, False,  self.info

    def seed(self, seed=None):
        pass
    def reset(self,seed=None, options=None):
        self.t = 0
        # 重置环境，返回初始状态
        self.done1 = False
        # 初始化无人机状态
        self.xy_p = [100, 100]
        self.xy_e = [600, 900]
        # 定义初始航向角
        self.heading_p = np.pi * 0.5
        self.heading_e = np.pi * 0.5
        self.drone_state_p[0] = self.get_angle2goal()
        self.drone_state_e[0] = self.get_angle2p()
        # 标准化状态空间
        for i in range(1):
            self.observation_e[i] = self.drone_state_e[i]
            self.observation_p[i] = self.drone_state_p[i]
            #self.observation[i] = (self.drone_state_p[i] - 0.5 * self.normalized[i]) / (0.5 * self.normalized[i])
            #.observation[i] = self.drone_state_p[i] / self.normalized[i]

        return self.observation_e, self.info

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
            #return (d_lod - d_new)*2 + self.step_penalty
            return -1*((d_lod - d_new) * 2 )+ self.step_penalty
    def get_angle(self):#返回航向角
        if self.heading_p > 2*np.pi:
            self.heading_p -= 2*np.pi

        return self.heading_p
    def get_d8(self, k):#返回8个测距仪的数据
        return 50
    def get_d2goal(self):#返回到目标点距离
        dx = self.xy_p[0] - self.xy_e[0]
        dy = self.xy_p[1] - self.xy_e[1]
        # 使用勾股定理计算距离
        distance = math.sqrt(pow(dx, 2) + pow(dy, 2))
        return distance
    def get_angle2goal(self):#返回到目标点的角度
        dx = self.xy_p[0] - self.xy_e[0]
        dy = self.xy_p[1] - self.xy_e[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        diff = (self.heading_p - angle_rad+ math.pi) % (2 * math.pi) - math.pi
        return diff
    def get_angle2p(self):#返回到追击者P的角度
        dx = self.xy_p[0] - self.xy_e[0]
        dy = self.xy_p[1] - self.xy_e[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        diff = (self.heading_e - angle_rad+ math.pi) % (2 * math.pi) - math.pi
        return diff
    def determine(self):

        if (self.get_d2goal() < self.r_goal):
            self.done1 = True
        if (self.t > 200):
            self.done1 = True
         #判断是否超出边界
        if (self.xy_p[0] < self.space1.low[0] or self.xy_e[0] < self.space1.low[0] or self.xy_e[0] > self.space1.high[0] or self.xy_p[0] > self.space1.high[0] or self.xy_e[1] < self.space1.low[1] or self.xy_e[1] > self.space1.high[1] or self.xy_p[1] < self.space1.low[1] or self.xy_p[1] > self.space1.high[1]):
            #print("222")
            self.done1 = True

    def _human_render(self):
        # Implement human-readable rendering logic here
        plt.scatter(self.xy_p[0], self.xy_p[1], marker='o', color='red', label='Drone_P', s=5)
        plt.scatter(self.xy_e[0], self.xy_e[1], marker='o', color='g', label='Drone_E', s=5)
        #plt.show()

    def _rgb_array_render(self):
        # Implement rendering logic to return an RGB array
        # This can be used for automated testing
        # (Similar to _human_render, but instead of displaying, return the RGB array)
        pass