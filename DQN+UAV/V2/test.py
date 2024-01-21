import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import env_v2
env = env_v2.DroneEnv()
# 在每个 episode 的开始调用一次 env.render()
state = env.reset()
env.render(mode='human')

model = DQN.load("C:/Users/10749/Desktop/my_UAV/DQN+UAV/V2/TrainedModel/Actor2.pkl")

done = False
r = 0
#plt.figure()
plt.scatter(env.goal[0], env.goal[1], marker='x', color='green', label='Goal')

# 创建一张图
#plt.figure()

# 绘制障碍物
for k in env.obstacles:
    obstacle_circle = plt.Circle(k, env.r_obstacles, color='lightblue', fill=True)
    plt.gca().add_patch(obstacle_circle)
plt.xlim(env.space1.low[0], env.space1.high[0])
plt.ylim(env.space1.low[1], env.space1.high[1])
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Drone Environment')
count = 0
while not done:
    count+=1
    print(count)

    env.render(mode='human')
    action, _states = model.predict(state, deterministic=True)
    #print(state, action)
    next_state, reward, done, info = env.step(action)
    if count > 300:
        done = True
    r += reward
    state = next_state
    #print(r)
plt.show()