import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
import env

env = env.DroneEnv()

# 创建一张图
plt.figure()

# 绘制障碍物
for k in env.obstacles:
    obstacle_circle = plt.Circle(k, env.r_obstacles, color='lightblue', fill=True)
    plt.gca().add_patch(obstacle_circle)

plt.scatter(env.goal[0], env.goal[1], marker='x', color='green', label='Goal')
plt.xlim(env.space1.low[0], env.space1.high[0])
plt.ylim(env.space1.low[1], env.space1.high[1])
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Drone Environment')
#plt.show()
state = env.reset()
env.render(mode='human')

model = DDPG.load("TrainedModel/Actor_Normalize_aw_cejv.pkl")

done = False
r = 0
count = 0
while not done:
    count += 1
    env.render(mode='human')
    action, _states = model.predict(state, deterministic=True)
    next_state, reward, done, info = env.step(action)
    print(state[0] , action, reward)
    if count > 500:
        done = True
    r += reward
    state = next_state

plt.show()
print(r)