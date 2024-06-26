import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import env_v3_P_TP
env = env_v3_P_TP.DroneEnv()
# 在每个 episode 的开始调用一次 env.render()
x=[]
model_p = DDPG.load("TrainedModel/Actor_p1.4.pkl")
for i in range(30):
    state,_ = env.reset()
    done = False
    r = 0
    t1 = 0
    while not done:
        t1 += 1
        #env.render(mode='human')
        action_p, _states_e = model_p.predict(state, deterministic=True)
        next_state, reward, done, t,  info = env.step(action_p)
        r += reward
        state = next_state
        #print(r)
    print(t1)
    x.append(t1)
# 添加标签和标题

plt.xlabel('Communication gap')
plt.ylabel('Successfully captured steps')
plt.legend()
plt.title('The relationship between success rate and interval')
plt.scatter(list(range(1, len(x) + 1)), x)
plt.show()