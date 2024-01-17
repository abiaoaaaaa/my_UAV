import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import os
import envlx_xiangdui
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env = envlx_xiangdui.DroneEnv()