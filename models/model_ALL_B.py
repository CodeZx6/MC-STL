import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import models.model_stmask as model_stmask
import models.GCN as GCN
import models.model_contrast as model_contrast
from helper.config import setup_seed
setup_seed(2021)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias= True)

class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn = False):
        super(_bn_relu_conv, self).__init__()
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        
        return x

class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual # short cut

        return out

class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x

class Contrast(nn.Module):
    def __init__(self, depth, in_chans, embed_dim, patch_size, img_size):
        super(Contrast, self).__init__()
        self.enc = model_contrast.contrast(depth=depth, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(256, 2)
        self.softmax = nn.Softmax()
        self.fusion = nn.Conv2d(2, 1, 3, 1, 1)

class Contrast_Enhance(nn.Module):
    def __init__(self, in_channels, out_channels,nb_residual_unit, patch_size, gru_dp,
                 map_height, map_width, len_X, embed_dim, depth,
                 base_channel=128, Beta=0.5):
        super(Contrast_Enhance, self).__init__()
        self.depth = depth
        self.len = len_X
        self.map_height = map_height
        self.map_width = map_width
        self.in_chans = in_channels
        self.patch_num = int(map_width/patch_size)**2
        self.embed_dim=embed_dim


        self.gru = nn.GRU(self.in_chans*self.map_width*self.map_height, self.in_chans*self.map_width*self.map_height, gru_dp, batch_first=True)

        self.contrast = Contrast(depth=self.depth, img_size=self.map_width, in_chans=self.in_chans*self.len, embed_dim=embed_dim, patch_size=patch_size)
        self.enc_c = self.contrast.enc.forward_encoder
        
        self.conv = nn.Conv2d(self.in_chans, self.len*self.in_chans, 3, 1, 1)
        
        self.linear = nn.Linear(in_features=self.patch_num*self.embed_dim, out_features=map_width*map_height*self.in_chans, bias=True)
        
        self.resnet = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=self.in_chans, out_channels=base_channel)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter=base_channel, repetations=nb_residual_unit)),
            ('Drop', nn.Dropout(0.3)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels=base_channel, out_channels=out_channels))
        ]))
    def forward(self, x):

        enc_output_vit = self.enc_c(x)
        enc_output_vit = self.linear(enc_output_vit.reshape(-1, self.patch_num*self.embed_dim))
        enc_output_vit = enc_output_vit.reshape(-1, self.in_chans, self.map_width, self.map_height)
        enc_output_vit = self.conv(enc_output_vit)
        x = enc_output_vit + x[:, :, :, :8]
        enc_output, h = self.gru(x.reshape(-1, self.len,  self.in_chans*self.map_height*self.map_width))
        enc_output = h[-1].reshape(-1, self.in_chans, self.map_width, self.map_height)
        enc_output = self.resnet(enc_output)
        return enc_output


class Mask_Enhance(nn.Module):
    def __init__(self, in_channels, out_channels, nb_residual_unit, patch_size,
                 map_height, map_width, len_X, embed_dim,
                 base_channel=128, Beta=0.5):
        super(Mask_Enhance, self).__init__()

        self.len = len_X
        self.map_height = map_height
        self.map_width = map_width
        self.in_chans = in_channels
        self.patch_num = int(map_width/patch_size)**2
        self.embed_dim=embed_dim

        self.vit = model_stmask.mask_vit(img_size=self.map_width, in_chans=self.in_chans*self.len, embed_dim=embed_dim, patch_size=patch_size)
        self.enc = self.vit.forward_encoder
        self.GCN = GCN.GCN_Precodition(len=self.len, map_width=self.map_width, patch_num=self.patch_num, embed_dim=self.embed_dim,in_chans=self.in_chans*self.len, Beta=Beta)
        
        self.conv = nn.Conv2d(int(self.in_chans/2), self.len, 3, 1, 1)
        self.linear = nn.Linear(in_features=self.patch_num*self.embed_dim, out_features=map_width*map_height*self.in_chans, bias=True)

        self.resnet = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=self.len*self.in_chans, out_channels=base_channel)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter=base_channel, repetations=nb_residual_unit)),
            ('Drop', nn.Dropout(0.3)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels=base_channel, out_channels=out_channels))
        ]))

        # self.end_conv = nn.Conv2d(4, 2, 3, 1, 1)
    def forward(self, x):

        enc_output, attn_qv = self.enc(x)
        enc_output = torch.unsqueeze(enc_output, dim=1)
        enc_output = self.conv(enc_output)
        gcn = self.GCN(x, attn_qv)
        enc_output += gcn

        enc_output = enc_output.reshape(-1, self.len, self.patch_num*self.embed_dim)
        enc_output = self.linear(enc_output).reshape(-1, self.len*self.in_chans, self.map_width, self.map_height)
        enc_output = self.resnet(enc_output)

        return enc_output


class STL(nn.Module):
    def __init__(self, in_channels, out_channels, nb_residual_unit, gru_dp, cdepth,
                 map_height, map_width, len_X, patch_size_ce, patch_size_me, embed_dim_ce, embed_dim_me,
                 base_channel=128, Beta=0.5):
        super(STL, self).__init__()

        self.ce = Contrast_Enhance(in_channels=in_channels, out_channels=out_channels, nb_residual_unit=nb_residual_unit,
                patch_size=patch_size_ce, gru_dp=gru_dp, depth=cdepth, map_height=map_height, map_width=map_width, len_X=len_X,
                embed_dim=embed_dim_ce, base_channel=base_channel, Beta=Beta)

        self.me = Mask_Enhance(in_channels=in_channels, out_channels=out_channels, nb_residual_unit=nb_residual_unit,
                patch_size=patch_size_me,map_height=map_height, map_width=map_width, len_X=len_X,
                embed_dim=embed_dim_me, base_channel=base_channel, Beta=Beta)
    def forward(self, x):
        
        cout = self.ce(x)
        mout = self.me(x)
        all_output =  cout + mout
        return all_output

class ext(nn.Module):
    def __init__(self):
        super(ext, self).__init__()
        self.embed_day = nn.Embedding(1232, 10)
        self.embed_weekend = nn.Embedding(2, 10)
        self.embed_hour = nn.Embedding(24, 10)  # hour range [0, 23]
        self.embed_holiday = nn.Embedding(2, 10)
        self.embed_wind = nn.Embedding(20, 10)
        self.embed_weather = nn.Embedding(18, 10)  # ignore 0, thus use 18
        self.embed_temperature = nn.Embedding(128, 10)

        self.exl = nn.Sequential(
            nn.Linear(60, 256),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024)
        )

    def forward(self, x):
        x = x.data.cpu().numpy()
        x = x[:, 1:]
        # ext_out1 = self.embed_day(torch.LongTensor(np.asarray(x[:, 0])).reshape(-1, 1).cuda()).view(-1, 3)
        ext_out1 = self.embed_weekend(torch.LongTensor(np.asarray(x[:, 0])).reshape(-1, 1).cuda()).view(-1, 10)
        ext_out2 = self.embed_hour(torch.LongTensor(np.asarray(x[:, 1])).reshape(-1, 1).cuda()).view(-1, 10)
        ext_out3 = self.embed_holiday(torch.LongTensor(np.asarray(x[:, 2])).reshape(-1, 1).cuda()).view(-1, 10)
        ext_out4 = self.embed_wind(torch.LongTensor(np.asarray(x[:, 3])).reshape(-1, 1).cuda()).view(-1, 10)
        ext_out5 = self.embed_weather(torch.LongTensor(np.asarray(x[:, 4])).reshape(-1, 1).cuda()).view(-1, 10)
        ext_out6 = self.embed_temperature(torch.LongTensor(np.asarray(x[:, 5])).reshape(-1, 1).cuda()).view(-1, 10)
        ext_out = self.exl(torch.cat([ext_out1, ext_out2, ext_out3, ext_out4, ext_out5, ext_out6], dim=1))
        return ext_out

class MC_STL(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_channels=64,
                 img_width=32, img_height=32, ext_flag=False, Beta=0.5, len_X = 10):
        super(MC_STL, self).__init__()
        patch_size_ce = 2
        patch_size_me = 2
        embed_dim_ce = 32
        embed_dim_me = 128
        cdepth = 2
        gru_dp = 2
        self.nb_residual_unit = 4
        self.img_width = img_width
        self.img_height = img_height
        self.ext_flag = ext_flag
        self.in_chan = in_channels

        if self.ext_flag == True:
            self.xc_ext = ext()
            # self.xp_ext = ext()
            # self.xt_ext = ext()

        self.stl = STL(in_channels= in_channels, out_channels=out_channels, nb_residual_unit=self.nb_residual_unit,
                                        map_height=self.img_height, map_width=self.img_width,
                                        Beta=Beta, len_X = len_X, base_channel=base_channels,
                                        patch_size_ce=patch_size_ce, patch_size_me=patch_size_me,
                                        embed_dim_ce=embed_dim_ce, embed_dim_me=embed_dim_me,
                                        gru_dp=gru_dp, cdepth=cdepth)

    def forward(self, X, X_ext=None):
        _, x_len, channel, _, _ = X.shape

        if self.ext_flag == True:
            X_ext = self.xc_ext(X_ext).view(-1, x_len, self.in_chan, self.img_width, self.img_height)
            X += X_ext

        X = X.view(-1, x_len * channel, self.img_width, self.img_height)
        x_zero = torch.zeros_like(X)
        X = torch.cat((x_zero, X), dim=-1)
        main_output = self.stl(X)

        return main_output
