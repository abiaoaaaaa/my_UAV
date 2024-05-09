# -*- coding: utf-8 -*-
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
from tensorflow.keras.models import Sequential, load_model
import pandas as pd
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Dropout
# 从 TensorFlow 中导入所需的模块
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import transbigdata as tbd
import warnings
warnings.filterwarnings("ignore")
np.random.seed(120)
tf.random.set_seed(120)
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from stable_baselines3 import DDPG
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DroneEnv(gym.Env):
    def __init__(self):
        #加载逃逸者模型
        self.model_TP = load_model("./model.h5")
        self.model_e = DDPG.load("TrainedModel/Actor_e1.4.pkl")
        self.t=0
        super(DroneEnv, self).__init__()
        # 定义状态空间和动作空间
        #状态空间x,y,v,航向角
        #self.d=50   #测距仪的范围
        self.observation_space = gym.spaces.Box(low=np.array([ -1*np.pi, -1*np.pi]), high=np.array([np.pi , np.pi]), dtype=np.float32)
        #self.observation_space = gym.spaces.Box(low=np.array([-1 * np.pi]), high=np.array([np.pi]), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([ -0.5*np.pi]), high=np.array([0.5*np.pi]), dtype=np.float32)
        #地图边界
        self.space1 = gym.spaces.Box(low=np.array([0, 0]), high=np.array([2000, 2000]), dtype=np.float32)
        #动作空间角速度
        #self.action_space = gym.spaces.Discrete(3)
        #self.change = [-0.2*np.pi, 0, 0.2*np.pi]#转向信号

        #UAV的坐标
        self.xy_p=[100,100]
        self.xy_e = [900, 900]
        # 定义初始航向角
        self.heading_p = 0.5*np.pi*0.5
        self.heading_e = 0.5 * np.pi * 0.5
        

        #匀速
        self.v_p = 20
        self.v_e = 10

        self.t_step = 1  # 时间步

        #状态空间，夹角
        self.drone_state_p = [np.pi]
        self.drone_state_e = [np.pi]
        #标准化用的两个数组
        # self.normalized = [2*np.pi, 1200, np.pi]
        self.observation_p = np.array([np.pi, np.pi],dtype=np.float32)
        #self.observation_p = np.array([np.pi], dtype=np.float32)
        self.observation_e = np.array([np.pi], dtype=np.float32)

        # 用于存储环境信息的字典
        self.info = {"collision": False}

        # 定义奖励函数中的常数
        self.xy_e_reward = 1000
        self.step_penalty = -26
        #self.step_penalty = 1
        self.r_goal = 50 #目标半径S
        self.done1 = False

        #定义训练TP的网络
        self.file_path = "train.csv"
        self.xy_e_next = [0, 0]
        self.id = 0
        self.data = np.array([[0 for _ in range(2)] for _ in range(10)], dtype='float64')

    def step(self, action):
        #self.save_data_to_csv(self.file_path)
        self.t += 1
        d_lod = self.get_d2goal()
        action_e, _states = self.model_e.predict(self.observation_e, deterministic=True)
        # 执行E动作
        self.heading_e += action_e  # 更新航向角
        self.xy_e[0] += self.v_e * self.t_step * math.cos(self.heading_e)  # 更新x坐标
        self.xy_e[1] += self.v_e * self.t_step * math.sin(self.heading_e)  # 更新y坐标
        # 执行P动作
        self.heading_p += action  # 更新航向角
        self.xy_p[0] += self.v_p * self.t_step * math.cos(self.heading_p)  # 更新x坐标
        self.xy_p[1] += self.v_p * self.t_step * math.sin(self.heading_p)  # 更新y坐标
        #预测E的下一个位置
        self.data = np.insert(self.data, self.data.shape[0], self.xy_e, axis=0)[1:]
        self.xy_e_next = self.model_TP(self.data.reshape(1, 10, 2)/2000)
        self.xy_e_next = self.xy_e_next[0]
        self.xy_e_next = self.xy_e_next*2000
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
            self.observation_p[i + 1] = self.get_angle2goal_next()
            #self.observation[i] = (self.drone_state_p[i] - 0.5 * self.normalized[i]) /(0.5 * self.normalized[i])
            #self.observation[i] = self.drone_state_p[i] / self.normalized[i]
        return self.observation_p, reward, self.done1, False,  self.info

    def seed(self, seed=None):
        pass
    def reset(self,seed=None, options=None):

        #self.save_data_to_csv(self.file_path)
        self.t = 0
        self.id += 1
        # 重置环境，返回初始状态
        self.done1 = False
        # 初始化无人机状态
        self.xy_p = [100, 100]
        self.xy_e = [900, 900]
        self.heading_p = 0
        self.heading_e = 0.5 * np.pi * 0.5
        self.drone_state_p[0] = self.get_angle2goal()
        self.drone_state_e[0] = self.get_angle2p()
        #初始化预测窗口
        for i in range(10):
            self.data = np.insert(self.data, self.data.shape[0], self.xy_e, axis=0)[1:]
        self.xy_e_next = self.model_TP(self.data.reshape(1, 10, 2) / 2000)
        self.xy_e_next = self.xy_e_next[0]
        self.xy_e_next = self.xy_e_next * 2000
        # 标准化状态空间
        for i in range(1):
            self.observation_e[i] = self.drone_state_e[i]
            self.observation_p[i] = self.drone_state_p[i]
            self.observation_p[i + 1] = self.get_angle2goal_next()
            #self.observation[i] = (self.drone_state_p[i] - 0.5 * self.normalized[i]) / (0.5 * self.normalized[i])
            #.observation[i] = self.drone_state_p[i] / self.normalized[i]

        return self.observation_p, self.info

    def save_data_to_csv(self, file_path):
        #print(111)
        # 如果文件不存在，则创建一个新文件，并写入列头
        if not os.path.isfile(file_path):
            #print(111)
            with open(file_path, 'w') as f:
                f.write(',id,step,e_x,e_y\n')
        # 将数据写入 CSV 文件
        with open(file_path, 'a') as f:
            f.write(f"{0},{self.id},{self.t + 1},{self.xy_e[0]},{self.xy_e[1]}\n")
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
    def get_angle2goal_next(self):#返回到预测的下一个目标点的角度
        dx = self.xy_p[0] - self.xy_e_next[0]
        dy = self.xy_p[1] - self.xy_e_next[1]
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