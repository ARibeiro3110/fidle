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
        print('init discriminator       : ',data_shape,' to sigmoid')

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