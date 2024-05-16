import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import env_test1

# 初始化环境
env = env_test1.DroneEnv()
#env = DummyVecEnv([lambda: env])

# 加载预训练模型
model = PPO.load("best_model")

# 设置绘图
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
circles = []  # 存储障碍物圆形对象

def draw_obstacles(obstacles):
    # 清除先前的障碍物绘制
    for circle in circles:
        circle.remove()
    circles.clear()

    # 绘制障碍物
    for k in obstacles:
        circle = plt.Circle(k, env.r_obstacles, color='lightblue', fill=True)
        ax.add_patch(circle)
        circles.append(circle)
    ax.set_xlim(env.space1.low[0], env.space1.high[0])
    ax.set_ylim(env.space1.low[1], env.space1.high[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('无人机环境')
    plt.draw()

# 绘制初始障碍物
draw_obstacles(env.obstacles)
env.render(mode='human')
# 模拟循环
done = False
state, _ = env.reset()
while not done:
    env.render(mode='human')
    action, _ = model.predict(state, deterministic=True)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    draw_obstacles(env.obstacles)
    plt.pause(0.001)  # 暂停一小段时间以更新绘图

plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终绘图
