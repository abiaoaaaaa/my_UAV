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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DroneEnv(gym.Env):
    def __init__(self):

        super(DroneEnv, self).__init__()
        # 定义状态空间和动作空间
        # 观察空间
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, -1]),
                                                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), dtype=np.float32)
        # 动作空间角速度,加速度
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        # 地图边界
        self.space1 = gym.spaces.Box(low=np.array([0, 0]), high=np.array([2000, 2000]), dtype=np.float32)
        # UAV的绝对坐标
        self.xy_p = [100, 100]

        # 定义起点、终点和障碍物
        self.heading_p = 0
        self.heading_e = 0
        self.xy_e = [600, 800]
        self.obstacles = []
        self.d = 300  # 测距仪的范围
        # 匀速
        self.v_p = 5
        self.v_e = 2

        self.t_step = 1  # 时间步
        self.count = 0
        self.MAX_count = 500
        # 状态空间
        # 第一个元素代表航向角的角度，八个测距仪的测量距离，倒数第二个元素代表了无人机到目标点的距离。最后一个元素代表了无人机当前朝向与目标点之间的角度差。
        # 真实状态
        self.drone_state = [0, 0, 0, 0, 0, 0, 0, 0, np.pi]
        # 标准化使用
        self.normalized = [self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d, np.pi]
        # 传递给模型的观察空间
        self.observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, np.pi], dtype=np.float32)
        # 用于存储环境信息的字典
        self.info = {"collision": False}
        self.truncated = False
        # 定义奖励函数中的常数
        self.xy_e_reward = 100
        self.step_penalty = -15
        self.d_penalty = 200
        self.obstacles_penalty = -1200

        self.r_goal = 20  # 目标半径
        self.r_obstacles = 100  # 障碍半径
        self.r_obstacles_threat = 100  # 障碍半径
        self.done1 = False

    def step(self, action):
        self.count += 1
        # 保存上一步的状态
        d_lod = self.get_d2goal()
        prev_state = np.array(self.drone_state)

        # 更新状态
        self.heading_p = (self.heading_p + action[0]) % (2 * np.pi)
        self.heading_e = self.get_angle2p()
        # self.v_p += action[1]  # 更新速度
        # self.v_p = min(self.v_p, 250)
        # self.v_p = max(0, self.v_p)
        self.xy_p[0] += self.v_p * self.t_step * math.cos(self.heading_p)  # 更新x坐标
        self.xy_p[1] += self.v_p * self.t_step * math.sin(self.heading_p)  # 更新y坐标

        # self.xy_e[0] += self.v_e * self.t_step * math.cos(self.heading_e)  # 更新x坐标
        # self.xy_e[1] += self.v_e * self.t_step * math.sin(self.heading_e)  # 更新y坐标

        # 更新测距仪的数据
        self.get_d8()
        self.drone_state[8] = self.get_angle2goal()
        # 判断是否到达终点
        self.determine()
        # 计算奖励
        reward = self.calculate_reward(d_lod)
        # 标准化观察空间
        for i in range(9):
            self.observation[i] = self.drone_state[i] / self.normalized[i]

        return self.observation, reward, self.done1, self.truncated, self.info

    def seed(self, seed=None):
        pass

    def reset(self, seed=0):
        # 重置环境，返回初始状态
        self.count = 0
        self.done1 = False
        self.truncated = False
        # 初始化无人机状态
        self.xy_p = [100, 100]
        self.heading_p = 0
        self.v_p = 5
        # 初始化障碍和目标
        # 随机生成目标点位置
        goal_radius = 400
        goal_center = [1200, 1200]
        goal_x = np.random.uniform(goal_center[0] - goal_radius, goal_center[0] + goal_radius)
        goal_y = np.random.uniform(goal_center[1] - goal_radius, goal_center[1] + goal_radius)
        self.xy_e = [goal_x, goal_y]
        # 随机生成两个障碍物位置
        obstacle_radius = 400
        obstacle_center = [800, 800]
        self.obstacles = []
        for _ in range(2):
            obstacle_x = np.random.uniform(obstacle_center[0] - obstacle_radius, obstacle_center[0] + obstacle_radius)
            obstacle_y = np.random.uniform(obstacle_center[1] - obstacle_radius, obstacle_center[1] + obstacle_radius)
            self.obstacles.append([obstacle_x, obstacle_y])
        self.get_d8()
        self.drone_state[8] = self.get_angle2goal()

        # 标准化状态空间
        for i in range(9):
            self.observation[i] = self.drone_state[i] / self.normalized[i]
        return self.observation, self.info

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

    def calculate_reward(self, d_lod):  # 计算奖励
        if (self.xy_p[0] < self.space1.low[0] or self.xy_p[0] > self.space1.high[0] or self.xy_p[1] < self.space1.low[
            1] or self.xy_p[1] > self.space1.high[1]):
            return -100
        d_new = self.get_d2goal()
        r3 = 0
        # r=50,d=100
        dismin = self.d_penalty
        for k in self.obstacles:
            dis = self.get_distance(self.xy_p, k)
            dismin = min(dis, self.d_penalty)
            if (dismin < self.d):
                r3 = self.obstacles_penalty + dismin * 6
                # 判断是否超出边界
        r4 = 0
        if (self.get_d2goal() < self.r_goal):
            r4 = self.xy_e_reward

        return (d_lod - d_new) * 2 + self.step_penalty + r3 + r4

    def get_angle(self):  # 返回航向角
        if self.heading_p > np.pi:
            self.heading_p -= 2 * np.pi
        elif self.heading_p < -np.pi:
            self.heading_p += 2 * np.pi
        return self.heading_p

    def get_d8(self):  # 返回8个测距仪的数据
        for i in range(8):
            self.drone_state[i] = 0
        for k in self.obstacles:
            dis = self.get_distance(self.xy_p, k)
            if (dis < self.d):
                angle = self.get_angle2obstacles(k)
                if (angle >= -0.5 * np.pi and angle < 0.5 * np.pi):
                    index = ((angle + 0.5 * np.pi) * 9) // (np.pi)
                    index = int(index)
                    if (self.drone_state[index] < self.d - dis):
                        self.drone_state[index] = self.d - dis
        '''new_indices = [3, 2, 1, 0, 7, 6, 5, 4]
        temp = self.drone_state
        for i, idx in enumerate(new_indices):
            self.drone_state[idx] = temp[i]'''

    def get_distance(self, a, b):  # 返回ab两点间距离
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        # 使用勾股定理计算距离
        distance = math.sqrt(pow(dx, 2) + pow(dy, 2))
        return distance

    def get_d2goal(self):  # 返回到目标点距离
        dx = self.xy_p[0] - self.xy_e[0]
        dy = self.xy_p[1] - self.xy_e[1]
        # 使用勾股定理计算距离
        distance = math.sqrt(pow(dx, 2) + pow(dy, 2))
        return distance

    def get_angle2goal(self):  # 返回到目标点的角度差
        dx = self.xy_p[0] - self.xy_e[0]
        dy = self.xy_p[1] - self.xy_e[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        diff = (self.heading_p - angle_rad + math.pi) % (2 * math.pi) - math.pi
        return diff

    def get_angle2p(self):  # 返回到追击者P的角度
        dx = self.xy_p[0] - self.xy_e[0]
        dy = self.xy_p[1] - self.xy_e[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        # diff = (self.heading_e - angle_rad+ math.pi) % (2 * math.pi) - math.pi
        # return diff
        return angle_rad

    def get_angle2obstacles(self, k):  # 返回到障碍点的角度
        dx = self.xy_p[0] - k[0]
        dy = self.xy_p[1] - k[1]
        # 计算角度（弧度）
        angle_rad = math.atan2(-dy, -dx)
        diff = (self.heading_p - angle_rad + math.pi) % (2 * math.pi) - math.pi
        return diff

    def determine(self):
        # 最大步数
        if self.count > self.MAX_count:
            self.done1 = True
            self.truncated = True
        # 是否到达
        if (self.get_d2goal() < self.r_goal):
            self.done1 = True
        # 是否碰到障碍
        # for k in self.obstacles:
        #     dis = self.get_distance(self.xy_p, k)
        #     if (dis<self.r_obstacles):
        #         self.done1 = True
        # 判断是否超出边界
        # if (self.xy_p[0] < self.space1.low[0] or self.xy_p[0] > self.space1.high[0] or self.xy_p[1] < self.space1.low[1] or self.xy_p[1] > self.space1.high[1]):
        # self.done1 = True

    def _human_render(self):
        # Implement human-readable rendering logic here
        plt.scatter(self.xy_p[0], self.xy_p[1], marker='o', color='red', label='Drone', s=0.1)
        plt.scatter(self.xy_e[0], self.xy_e[1], marker='o', color='g', label='Drone_E', s=0.1)
        # plt.show()

    def _rgb_array_render(self):
        # Implement rendering logic to return an RGB array
        # This can be used for automated testing
        # (Similar to _human_render, but instead of displaying, return the RGB array)
        pass