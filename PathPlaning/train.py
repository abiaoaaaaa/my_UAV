from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import gym
import numpy as np

import env
env = env.DroneEnv()
#check_env(env)
# 把环境向量化，如果有多个环境写成列表传入DummyVecEnv中，可以用一个线程来执行多个环境，提高训练效率
envs = DummyVecEnv([lambda: env])
vec_env = VecNormalize(envs, norm_obs=True, norm_reward=True,
                   clip_obs=0.03)
# 定义一个DDPG模型，设置其中的各个参数
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
print(n_actions)
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.006 * np.ones(n_actions))

model = DDPG("MlpPolicy", vec_env, verbose=1, action_noise=action_noise, tensorboard_log="./tensorboard/",
             device="cuda")
# 开始训练
model.learn(total_timesteps=100000, log_interval=10)
# 策略评估
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
#env.close()
print("mean_reward:",mean_reward,"std_reward:",std_reward)
# 保存模型到相应的目录
model.save("C:/Users/10749/Desktop/my_UAV/PathPlaning/TrainedModel/Actor_Normalize_aw_cejv.pkl")

