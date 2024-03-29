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
        self.d=200   #测距仪的范围
        self.observation_space = gym.spaces.Box(low=np.array([-1*np.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1*np.pi]), high=np.array([np.pi,self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, np.pi]), dtype=np.float32)
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
        self.obstacles = [[500,500],[200,200],[300,400],[400,300],[800,800], [400,200]]

        #匀速
        self.v = 20
        self.t_step = 1  # 时间步
        self.count = 0
        self.MAX_count = 500
        #状态空间

        #真实状态
        self.drone_state = [0,self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, np.pi]
        #标准化使用
        self.normalized = [2*np.pi, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, 1200, 2*np.pi]
        #传递给模型的观察空间
        self.observation = [0,self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, np.pi]


        # 用于存储环境信息的字典
        self.info = {"collision": False}

        # 定义奖励函数中的常数
        self.goal_reward = 1000
        self.step_penalty = -6
        self.obstacles_penalty = -20

        self.r_goal = 50  # 目标半径
        self.r_obstacles = 50  # 障碍半径
        self.done1 = False
    def step(self, action):
        self.count +=1
        # 保存上一步的状态
        d_lod = self.get_d2goal()
        prev_state = np.array(self.drone_state)

        # 更新状态
        self.heading += self.change[action]# 更新航向角
        self.xy[0] += self.v * self.t_step * math.cos(self.heading) # 更新x坐标
        self.xy[1] += self.v * self.t_step * math.sin(self.heading)  # 更新y坐标
        self.drone_state[0] = self.get_angle()
        #更新测距仪的数据

        self.get_d8()
        self.drone_state[10] = self.get_d2goal()
        self.drone_state[11] = self.get_angle2goal()

        # 判断是否到达终点
        self.determine()


        # 计算奖励
        reward = self.calculate_reward(d_lod)
        #标准化观察空间
        for i in range(12):
            self.observation[i] = self.drone_state[i]
            #self.observation[i] = (self.drone_state[i] - 0.5 * self.normalized[i]) /(0.5 * self.normalized[i])
            #self.observation[i] = self.drone_state[i] / self.normalized[i]
        return self.observation, reward, self.done1, self.info


    def reset(self):
        # 重置环境，返回初始状态
        self.count = 0
        self.done1 = False
        # 初始化无人机状态
        self.xy = [100, 100]
        self.heading = 0
        self.drone_state[0] = self.get_angle()

        self.get_d8()
        self.drone_state[10] = self.get_d2goal()
        self.drone_state[11] = self.get_angle2goal()
        # 标准化状态空间
        for i in range(12):
            self.observation[i] = self.drone_state[i]
            #self.observation[i] = (self.drone_state[i] - 0.5 * self.normalized[i]) / (0.5 * self.normalized[i])
            #self.observation[i] = self.drone_state[i] / self.normalized[i]

        return self.observation

    def render(self, mode='human'):
        # 可选的渲染方法，用于可视化环境
        if mode == 'human':
            # Human-readable rendering (e.g., visual display)
            self._human_render()
        elif mode == 'rgb_array':
            # Return an RGB array for rendering in automated testing
            return self._rgb_array_render()
        else:
            super().render(mode=mode)
        pass
    def calculate_reward(self, d_lod):# 计算奖励
        d_new = self.get_d2goal()
        r3 = 0
        for k in self.obstacles:
            dis = self.get_distance(self.xy,k)
            if (dis<self.r_obstacles):
                r3=self.obstacles_penalty
        return (d_lod - d_new) * 2 + self.step_penalty + r3
    def get_angle(self):#返回航向角
        if self.heading > 2*np.pi:
            self.heading -= 2*np.pi

        return self.heading
    def get_d8(self):#返回8个测距仪的数据
        for i in  range(9):
            self.drone_state[1+i] = self.d
        for k in self.obstacles:
            dis = self.get_distance(self.xy,k)
            if (dis<self.d):
                angle = self.get_angle2obstacles(k)
                index = ((angle + np.pi)*9)//(2*np.pi)
                index = int(index)
                if (self.drone_state[1 + index] > dis) :
                    self.drone_state[1 + index] = dis






    def get_distance(self,a,b):#返回ab两点间距离
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        # 使用勾股定理计算距离
        distance = math.sqrt(pow(dx, 2) + pow(dy, 2))
        return distance
    def get_d2goal(self):#返回到目标点距离
        dx = self.xy[0] - self.goal[0]
        dy = self.xy[1] - self.goal[1]
        # 使用勾股定理计算距离
        distance = math.sqrt(pow(dx, 2) + pow(dy, 2))
        return distance
    def get_angle2goal(self):#返回到目标点的角度差
        dx = self.xy[0] - self.goal[0]
        dy = self.xy[1] - self.goal[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        diff = (self.heading - angle_rad + math.pi) % (2 * math.pi) - math.pi
        return diff

    def get_angle2obstacles(self,k):  # 返回到障碍点的角度
        dx = self.xy[0] - k[0]
        dy = self.xy[1] - k[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        diff = (self.heading - angle_rad + math.pi) % (2 * math.pi) - math.pi
        return diff
    def determine(self):
        if self.count >self.MAX_count:
            self.done1 = True
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