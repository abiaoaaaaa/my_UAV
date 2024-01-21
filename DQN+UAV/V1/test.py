import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import env_v1
env = env_v1.DroneEnv()
# 在每个 episode 的开始调用一次 env.render()
state = env.reset()
env.render(mode='human')

model = DQN.load("C:/Users/10749/Desktop/my_UAV/DQN+UAV/V1/TrainedModel/Actor1.pkl")

done = False
r = 0
#plt.figure()
plt.scatter(env.goal[0], env.goal[1], marker='x', color='green', label='Goal')
plt.xlim(env.space1.low[0], env.space1.high[0])
plt.ylim(env.space1.low[1], env.space1.high[1])
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Drone Environment')
while not done:
    env.render(mode='human')
    action, _states = model.predict(state, deterministic=True)
    #print(state, action)
    next_state, reward, done, info = env.step(action)
    r += reward
    state = next_state
    #print(r)
plt.show()