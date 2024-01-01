from __future__ import print_function
import os
import platform

import torch
import numpy as np
import random
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
setup_seed(2021)
class Config(object):
    """docstring for Config"""
    DATAPATH = 'D:\Develop\STC1\data'
    def __init__(self):
        super(Config, self).__init__()

        #DATAPATH = ' '
