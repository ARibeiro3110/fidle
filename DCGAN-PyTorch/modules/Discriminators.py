# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                         GAN / Generators
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG MIAI/EFELIA 2023 - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# by JL Parouty (feb 2023) - PyTorch Lightning example

import numpy as np
import torch.nn as nn

class Discriminator_1(nn.Module):

    def __init__(self, latent_dim=None, data_shape=None):
    
        super().__init__()
        self.img_shape = data_shape
        print('init discriminator 1     : ',data_shape,' to sigmoid')

        self.model = nn.Sequential(

            nn.Flatten(),
            nn.Linear(int(np.prod(data_shape)), 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # img_flat = img.view(img.size(0), -1)
        validity = self.model(img)

        return validity




class Discriminator_2(nn.Module):

    def __init__(self, latent_dim=None, data_shape=None):
    
        super().__init__()
        self.img_shape = data_shape
        print('init discriminator 2     : ',data_shape,' to sigmoid')

        self.model = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(12544, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_nchw = img.permute(0, 3, 1, 2) # from NHWC to NCHW
        validity = self.model(img_nchw)

        return validity