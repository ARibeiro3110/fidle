
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


class Generator_1(nn.Module):

    def __init__(self, latent_dim=None, data_shape=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape  = data_shape
        print('init generator           : ',latent_dim,' to ',data_shape)

        self.model = nn.Sequential(
            
            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128,256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.ReLU(),

            nn.Linear(1024, int(np.prod(data_shape))),
            nn.Sigmoid()

        )


    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img