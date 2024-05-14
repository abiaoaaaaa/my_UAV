import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import env_v3_P_TP
env = env_v3_P_TP.DroneEnv()
# 在每个 episode 的开始调用一次 env.render()
ls=[]
ls1=[]
for i in range(10000):
    step = 0
    state,_ = env.reset()
    #env.render(mode='human')
    model_p = DDPG.load("TrainedModel/Actor_p1.4.pkl")
    done = False
    while not done:
        step+=1
        #env.render(mode='human')
        action_p, _states_p = model_p.predict(state, deterministic=True)
        next_state, reward, done, t,  info = env.step(action_p)
        state = next_state
    ls.append(i)
    ls1.append(step)
plt.scatter(ls,ls1)
plt.show()

