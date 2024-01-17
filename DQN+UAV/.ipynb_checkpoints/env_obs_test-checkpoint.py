from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import math
import random
import gym
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
        self.observation_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([150,150]),dtype=np.float32)
        #动作空间加速度、角速度
        self.action_space = gym.spaces.Discrete(4)

        # 定义起点、终点和障碍物
        self.start = (0,0)  # 初始状态：x, y, 速度，角度
        self.goal = (100, 100)
        self.obstacles = [(20, 40)]

        # 初始化无人机状态
        self.drone_state =  [float(x) for x in np.array(self.start)]
        # 定义奖励函数中的常数
        self.goal_reward = 100
        self.collision_penalty = -50
        self.Border_penalty = -100
        self.step_penalty = -1
        self.t_step = 1 #时间步
        self.r_obstacles = 5#障碍半径
        self.r_goal = 3 #目标半径
        self.done1 = False
        self.change = [[0, -1], [0, 1], [-1, 0], [1, 0]]#下、上、左、右
        
    def step(self, action):
        # 保存上一步的状态
        prev_state = np.array(self.drone_state)
        #print(type(self.drone_state[0]))
        # 更新状态

        self.drone_state[0] += self.change[action][0] # 更新x坐标
        self.drone_state[1] += self.change[action][1] # 更新y坐标

        # 判断是否到达终点
        if (abs(self.drone_state[0] - self.goal[0]) < self.r_goal) and  (abs(self.drone_state[1] - self.goal[1]) < self.r_goal):
            #print("111")
            self.done1 = True
            
        #判断是否碰到障碍
        if (abs(self.drone_state[0] - self.obstacles[0][0]) < self.r_obstacles) and  (abs(self.drone_state[1] - self.obstacles[0][1]) < self.r_obstacles):
            self.done1 = True
            
            #判断是否超出边界
        if (self.drone_state[0] < self.observation_space.low[0] or self.drone_state[0] > self.observation_space.high[0] or self.drone_state[1] < self.observation_space.low[1] or self.drone_state[1] > self.observation_space.high[1]):
            #print("222")
            self.done1 = True
            
        # 计算奖励
        reward = self.calculate_reward(prev_state)
    ############################################################################
        #print(action,self.drone_state, reward,self.done1) 
        return self.drone_state, reward, self.done1 , action

    def calculate_reward(self,prev_state): # 计算奖励
        
        #判断是否终点
        if (abs(self.drone_state[0] - self.goal[0]) < self.r_goal) and  (abs(self.drone_state[1] - self.goal[1]) < self.r_goal):
            return self.goal_reward
        
        #判断是否碰到障碍
        if (abs(self.drone_state[0] - self.obstacles[0][0]) < self.r_obstacles) and  (abs(self.drone_state[1] - self.obstacles[0][1]) < self.r_obstacles):
            return self.collision_penalty
        
        #判断是否超出边界
        if (self.drone_state[0] < self.observation_space.low[0] or self.drone_state[0] > self.observation_space.high[0] or self.drone_state[1] < self.observation_space.low[1] or self.drone_state[1] > self.observation_space.high[1]):
            return self.Border_penalty

        else:
            d_new = (self.drone_state[0]-self.goal[0])**2 + (self.drone_state[1]-self.goal[1])**2
            d_lod =  (prev_state[0]-self.goal[0])**2 + (prev_state[1]-self.goal[1])**2
            #指数奖励
            #print(d_new ** 0.5 , d_lod**0.5 - d_new**0.5)
            return d_lod**0.5 - d_new**0.5 - 1
            #if(d_lod**0.5 - d_new**0.5 > 0): #正负奖励
              #  return 1
            #else:
            #    return -1
            #return d_lod**0.5 - d_new**0.5 -125   #距离奖励 

    def reset(self):
        # 重置环境，返回初始状态
        self.drone_state = self.drone_state =  [float(x) for x in np.array(self.start)]
        self.done1 = False
        return self.drone_state

    def render(self):
        # 可选的渲染方法，用于可视化环境
        pass