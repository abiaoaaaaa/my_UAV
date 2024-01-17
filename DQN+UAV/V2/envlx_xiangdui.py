from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import math
import random
import gym

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        # 定义状态空间和动作空间
        #状态空间x,y,v,航向角
        self.d=100
        self.observation_space = gym.spaces.Box(low=np.array([-1*np.pi,0, 0, 0, 0, 0, 0, 0, 0, 0, -1*np.pi]), high=np.array([np.pi,self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, np.pi]), dtype=np.float32)
        #动作空间角速度
        self.action_space = gym.spaces.Discrete(5)
        self.change = [-1*np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]#转向信号


        # 定义起点、终点和障碍物
        self.start = [100, 100]  # 初始状态：x, y
        self.goal = [900, 900]
        self.heading = 0

        #self.obstacles = [(10, 90)]
        self.drone_state = [0,self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, np.pi]
        # 初始化无人机状态
        self.drone_state[0] = self.get_angle( )
        for i in range(1,10):
            self.drone_state[i] =  self.get_d8(i)
        self.drone_state[10] = self.get_d2goal( )
        self.drone_state[11] = self.get_angle2goal( )

        # 定义奖励函数中的常数
        self.goal_reward = 10000
        self.collision_penalty = -10000
        self.step_penalty = -1
        self.t_step = 1 #时间步
        self.r_obstacles = 100#障碍半径
        self.r_goal = 1000 #目标半径S
        self.done1 = False
    def get_angle(self):#返回航向角
        return 1
    def get_d8(self, k):#返回8个测距仪的数据
        return 1
    def get_d2goal(self):#返回到目标点距离
        return 1
    def get_angle2goal(self):#返回到目标点的角度
        return 1
    def step(self, action):
        # 保存上一步的状态
        prev_state = np.array(self.drone_state)
        #print(type(self.drone_state[0]))
        # 更新状态
        self.drone_state[3] += self.change[action] * self.t_step# 更新角度
        self.drone_state[0] -= (self.drone_state[2] * self.t_step) * math.cos(self.drone_state[3]*0.001) # 更新x坐标
        self.drone_state[1] -= (self.drone_state[2] * self.t_step) * math.sin(self.drone_state[3]*0.001) # 更新y坐标
        #0.01self.drone_state[2] += action[0] * self.t_step  # 更新速度
        #print(self.drone_state)


        # 判断是否到达终点
        if (self.drone_state[0] ** 2 + self.drone_state[1]  ** 2) ** 0.5 < self.r_goal:
            #print("111")
            self.done1 = True
         #判断是否超出边界
        if (self.goal[0] - self.drone_state[0] < self.observation_space.low[0] or self.goal[0] - self.drone_state[0] > self.observation_space.high[0] or self.goal[1] - self.drone_state[1] < self.observation_space.low[1] or self.goal[1] - self.drone_state[1] > self.observation_space.high[1]):
            #print("222")
            self.done1 = True
        # 计算奖励
        reward = self.calculate_reward(prev_state)
    ############################################################################
        #print(  np.round(prev_state[0:2],2),np.round(prev_state[3]*100,2),np.round(self.change[action]*100,2), np.round(self.drone_state[0:2],2),reward )
        return self.drone_state, reward, self.done1,action
    
    def calculate_reward(self,prev_state):# 计算奖励
        #判断是否超出边界
        if (self.goal[0] - self.drone_state[0] < self.observation_space.low[0] or self.goal[0] - self.drone_state[0] > self.observation_space.high[0] or self.goal[1] - self.drone_state[1] < self.observation_space.low[1] or self.goal[1] - self.drone_state[1] > self.observation_space.high[1]):
            return self.collision_penalty
            
        # 判断是否到达终点
        if (self.drone_state[0] ** 2 + self.drone_state[1]  ** 2) ** 0.5 < self.r_goal:
            return self.goal_reward
        else:
            d_new = (self.drone_state[0])**2 + (self.drone_state[1])**2
            d_lod =  (prev_state[0])**2 + (prev_state[1])**2
            #指数奖励
            #print(d_new ** 0.5 , d_lod**0.5 - d_new**0.5)
            return d_lod**0.5 - d_new**0.5 -150
            #if(d_lod**0.5 - d_new**0.5 > 0): #正负奖励
              #  return 1
            #else:
            #    return -1
            #return d_lod**0.5 - d_new**0.5 -125   #距离奖励 

    def reset(self):
        # 重置环境，返回初始状态
        self.done1 = False
        # 初始化无人机状态
        self.drone_state[0] = self.get_angle()
        for i in range(1, 10):
            self.drone_state[i] = self.get_d8(i)
        self.drone_state[10] = self.get_d2goal()
        self.drone_state[11] = self.get_angle2goal()

    def render(self):
        # 可选的渲染方法，用于可视化环境
        pass