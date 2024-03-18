#  V1.3（使用DDPG连续动作空间，无障碍，目标点是动态移动的,并且训练逃逸者E）ing
## 环境配置
![img.png](image/env_v1.png)
## 运行结果
![img_1.png](image/outcome1.png)
![img_1.png](image/outcome2.png)

## 动作空间change = [-0.5*np.pi,0.5*np.pi]#转向信号
## 原有的文件结构env_v3,train
## 训练逃逸者的文件结构env_v3_e,train_e,env_v3_e里面实现对追逃者的动作选择和执行
