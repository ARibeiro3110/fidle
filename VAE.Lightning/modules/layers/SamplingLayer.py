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
# by Achille Mbogol Touye (sep 2023), based on https://www.researchgate.net/publication/304163568_Tutorial_on_Variational_Autoencoders
#

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SamplingLayer(nn.Module):
    '''A custom layer that receive (z_mean, z_var) and sample a z vector'''

    def forward(self, inputs):
        
        z_mean, z_logvar = inputs
        
        batch_size = z_mean.size(0)
        latent_dim = z_mean.size(1)

        z_sigma    = torch.exp(0.5 * z_logvar)
        
        epsilon    = torch.randn(size=(batch_size, latent_dim)).to(device)  
        
        z          = z_mean + z_sigma * epsilon
        
        return z