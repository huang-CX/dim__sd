import torch
import numpy as np
# Hyper Parameters:
sequence_len = 3
BATCH_SIZE = 128
state_dim = 23
action_dim = 5
reward_dim = 5
pi = 3.1415926
seed = 56
position = (1.0, 0.0, 0.0)
USE_CUDA = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
