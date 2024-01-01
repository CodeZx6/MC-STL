import os
import sys
import numpy as np
import argparse
import warnings
from helper.config import Config, setup_seed
import torch
import torch.nn as nn
from helper.preprocessing import MinMaxNormalization
from helper.utils.metrics import get_MSE, get_MAE
from helper.make_dataset import get_dataloader, print_model_parm_nums
from utils import weights_init_normal
from models.model_ALL import MC_STL
sys.path.append('.')
warnings.filterwarnings('ignore')
# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--base_channels', type=int, default=128, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=32, help='image width')
parser.add_argument('--img_height', type=int, default=32, help='image height')
parser.add_argument('--channels', type=int, default=2, help='number of flow image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between validation')
parser.add_argument('--harved_epoch', type=int, default=50, help='halved at every x interval')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--ext_flag', type=bool, default=True, help='external factors')
parser.add_argument('--dataset', type=str, default='TaxiBJ_P1', help='which dataset to use: TaxiBJ, TaxiBJ_P1 etc.')
parser.add_argument('--change_epoch', type=int, default=0, help='change optimizer')
parser.add_argument('--len_previous', type=int, default=10, help='Length of historical traffic')
parser.add_argument('--Beta', type=float, default=0.5, help='c_flag')
opt = parser.parse_args()
print(opt)
setup_seed(opt.seed)

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.backends.cudnn.benchmark = True

# initial model
model = MC_STL( in_channels=opt.channels,
            out_channels=opt.channels,  
            img_width=opt.img_width,
            img_height=opt.img_height,
            base_channels=opt.base_channels,
            ext_flag=opt.ext_flag,
            Beta = opt.Beta,
            len_X =  opt.len_previous
            )
model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'MC-STL')

criterion = nn.MSELoss()
if cuda:
    model.cuda()
    criterion.cuda()

# load test set
datapath = os.path.join('./data', opt.dataset)
test_dataloader, max_min= get_dataloader(datapath, opt.batch_size, 'test')
mmn = MinMaxNormalization()
mmn._max = max_min.item()[max]
mmn._min = max_min.item()[min]

model_path = os.path.join('./Saved_models', opt.dataset)
model.load_state_dict(torch.load('{}/best_model.pt'.format(model_path)))
model.eval()
total_mse, total_mae = 0, 0
for j, (X, X_ext, Y) in enumerate(test_dataloader):
    preds = model(X, X_ext)
    preds = mmn.inverse_transform(preds).cpu().detach().numpy()
    Y = mmn.inverse_transform(Y).cpu().detach().numpy()
    total_mse += get_MSE(preds, Y) * len(Y)
    total_mae += get_MAE(preds, Y) * len(Y)
rmse = np.sqrt(total_mse / len(test_dataloader.sampler))
mae = total_mae / len(test_dataloader.sampler)
print("Test Result\t RMSE: {:.6f}\tMAE: {:.6f}\t".format(rmse, mae))