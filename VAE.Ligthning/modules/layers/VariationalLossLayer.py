# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                            SamplingLayer
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 
# ------------------------------------------------------------------
# by Achille Mbogol Touye (sep 2020), based on https://www.researchgate.net/publication/304163568_Tutorial_on_Variational_Autoencoders

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalLossLayer(nn.Module):
   
    def __init__(self, loss_weights=[3,7]):
        super().__init__()
        self.k1 = loss_weights[0]
        self.k2 = loss_weights[1]


    def forward(self, inputs):
        
        # ---- Retrieve inputs
        #
        x, z_mean, z_logvar, x_hat = inputs
        
        # ---- Compute : reconstruction loss
        #
        r_loss  = F.mse_loss(x_hat, x)* self.k1
        
        #
        # ---- Compute : kl_loss
        #
        kl_loss =  - torch.mean(1 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar))* self.k2

        # ---- total loss
        #
        loss   = r_loss + kl_loss
       
        return r_loss, kl_loss, loss

    
    def get_config(self):
        return {'loss_weights':[self.k1,self.k2]}