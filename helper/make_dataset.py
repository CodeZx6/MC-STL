import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from helper.config import setup_seed
setup_seed(2021)

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))

def get_dataloader(datapath, batch_size, mode='train'):

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    max_min = np.load(os.path.join(datapath, 'mm.npy'), allow_pickle=True)
    datapath = os.path.join(datapath, mode)
    X = np.load(os.path.join(datapath, 'time_correlation.npy'))
    X_ext = np.load(os.path.join(datapath, 'time_c_feature.npy'))
    Y = np.load(os.path.join(datapath, 'basis.npy'))
    

    X_ext = X_ext.astype(np.float64)
    X_ext = Tensor(X_ext)
    Y = Tensor(Y)
    X = Tensor(X)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, X_ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return dataloader, max_min

def get_dataloader_Bike(datapath, batch_size, mode='train'):

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    max_min = np.load(os.path.join(datapath, 'mm.npy'), allow_pickle=True)
    datapath = os.path.join(datapath, mode)
    X = np.load(os.path.join(datapath, 'time_correlation.npy'))
    Y = np.load(os.path.join(datapath, 'basis.npy'))
    
    Y = Tensor(Y)
    X = Tensor(X)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return dataloader, max_min