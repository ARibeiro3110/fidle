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

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear=nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 64*7*7),
            nn.BatchNorm1d(64*7*7),
            nn.ReLU()
        )
        
        self.Deconvblock=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=1,  kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    

    def forward(self, z):
       x        = self.linear(z)
       x        = x.reshape(-1,64,7,7)
       x_hat    = self.Deconvblock(x)
       return x_hat
        