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
import lightning.pytorch as pl

from IPython.display import display,Markdown
from modules.layers  import VariationalLossLayer


class VAE(pl.LightningModule):
    '''
    A VAE model, built from given encoder and decoder
    '''

    version = '1.4'

    def __init__(self, encoder=None, decoder=None, loss_weights=[1,.001], **kwargs):
        '''
        VAE instantiation with encoder, decoder and r_loss_factor
        args :
            encoder : Encoder model
            decoder : Decoder model
            loss_weights : Weight of the loss functions: reconstruction_loss and kl_loss
            r_loss_factor : Proportion of reconstruction loss for global loss (0.3)
        return:
            None
        '''
        super(VAE, self).__init__(**kwargs)
        self.encoder      = encoder
        self.decoder      = decoder
        self.loss_weights = loss_weights
        print(f'Fidle VAE is ready :-)  loss_weights={list(self.loss_weights)}')
       
        
    def forward(self, inputs):
        '''
        args:
            inputs : Model inputs
        return:
            output : Output of the model 
        '''
        z_mean, z_logvar, z = self.encoder(inputs)
        output              = self.decoder(z)
        return output
                    

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        inputs, _           = batch
        z_mean, z_logvar, z = self.encoder(inputs)
        x_hat               = self.decoder(z)
        
        r_loss,kl_loss,loss = VariationalLossLayer(loss_weights=self.loss_weights)([inputs, z_mean,z_logvar,x_hat]) 

        metrics = { "r_loss"     : r_loss, 
                    "kl_loss"    : kl_loss,
                    "vae_loss"   : loss
                  }
        
        # logs metrics for each training_step
        self.log_dict(metrics,
                      on_step  = False,
                      on_epoch = True, 
                      prog_bar = True, 
                      logger   = True
                     ) 
        
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        

    
    @classmethod
    def about(cls):
        '''Basic whoami method'''
        display(Markdown('<br>**FIDLE 2023 - VAE**'))
        print('Version              :', cls.version)
        print('Lightning version    :', pl.__version__)
