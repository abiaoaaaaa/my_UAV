import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import env_v3_e
env = env_v3_e.DroneEnv()
# 在每个 episode 的开始调用一次 env.render()
state,_ = env.reset()
env.render(mode='human')


#model_p = DDPG.load("TrainedModel/Actor1.2.pkl")
model_e = DDPG.load("TrainedModel/Actor_e1.4.pkl")
t1 = 0
done = False
r = 0
#plt.figure()
#plt.scatter(env.goal[0], env.goal[1], marker='x', color='green', label='Goal')
plt.xlim(env.space1.low[0], env.space1.high[0])
plt.ylim(env.space1.low[1], env.space1.high[1])
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Drone Environment')
print(state)
while not done:
    t1 += 1
    env.render(mode='human')
    #action_p, _states_p = model_p.predict(state, deterministic=True)
    action_e, _states_e = model_e.predict(state, deterministic=True)
    #action_e = 0
    #print(action)
    print(state, action_e)
    next_state, reward, done, t,  info = env.step(action_e)
    r += reward
    state = next_state
    print(r)
print(t1)
plt.show()