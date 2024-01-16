# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                              VAE Example
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 
# ------------------------------------------------------------------
# by Achille Mbogol Touye (sep 2023)
#

import os
import torch
import numpy as np
import torch.nn as nn
from modules.layers  import SamplingLayer

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.Convblock=nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),

            nn.Linear(64*7*7, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
        )

        self.z_mean   = nn.Linear(16, latent_dim)
        self.z_logvar = nn.Linear(16, latent_dim)
        


    def forward(self, x):
       x        = self.Convblock(x)
       z_mean   = self.z_mean(x)
       z_logvar = self.z_logvar(x) 
       z        = SamplingLayer()([z_mean, z_logvar]) 
         
       return z_mean, z_logvar, z 
