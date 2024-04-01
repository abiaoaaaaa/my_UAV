# my_UAV无障碍版本(done)DDPG
## 环境配置
![img.png](image/env_v1.png)
## 运行结果
![img_1.png](image/outcome1.png)
![img_1.png](image/outcome2.png)
## 泛化分析，目标点设置300，900.
![img_1.png](image/fanhua1.png)
## 动作空间change = [-0.5*np.pi,0.5*np.pi]#转向信号

## 3.测试观测空间扩大1000倍是否影响，即（判断 是否需要归一化，实验结果证明确实需要归一化，重要的信息要保留，不可以被“遮挡”）
### 3.1 观测空间扩大1000倍[angel*1000]
![obs_1000.png](image%2Fobs_1000.png)
![img.png](image/img4.png)
### 3.2 观测空间改为[angel,900]
![obs_pi_900.png](image%2Fobs_pi_900.png)
![img.png](image/img.png)
### 3.3 参考标准的观测空间[angel]
![obs_1.png](image%2Fobs_1.png)
![img.png](image/img3.png)

### 3.4 观测空间改为[pi,0]
![img.png](image/img2.png)
![img.png](image/img1.png)