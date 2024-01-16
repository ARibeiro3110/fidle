# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                            ImageCallback
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 
# ------------------------------------------------------------------
# 2.0 version by Achille Mbogol Touye, sep 2023


import lightning.pytorch as pl
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImagesCallback(pl.Callback):
    '''
    Save generated (random mode) or encoded/decoded (z mode) images on epoch end.
    params:
        x           : input images, for z mode (None)
        z_dim       : size of the latent space, for random mode (None)
        nb_images   : number of images to save
        from_z      : save images from z (False)
        from_random : save images from random (False)
        filename    : images filename
        run_dir     : output directory to save images        
    '''
    
   
    def __init__(self, x           = None,
                       z_dim       = None,
                       nb_images   = 5,
                       from_z      = False, 
                       from_random = False,
                       filename    = 'image-{epoch:03d}-{i:02d}.jpg',
                       run_dir     = './run'):
        
        # ---- Parameters
        #import lightning.pytorch as pl
        super().__init__()
        self.x = None if x is None else x[:nb_images]
        self.z_dim       = z_dim
        
        self.nb_images   = nb_images
        self.from_z      = from_z
        self.from_random = from_random

        self.filename_z       = run_dir + '/images-z/'      + filename
        self.filename_random  = run_dir + '/images-random/' + filename
        
        if from_z:      os.makedirs( run_dir + '/images-z/',      mode=0o750, exist_ok=True)
        if from_random: os.makedirs( run_dir + '/images-random/', mode=0o750, exist_ok=True)
        
    
    
    def save_images(self, images, filename, epoch):
        '''Save images as <filename>'''
        
        for i,image in enumerate(images):
            
            image = image.squeeze()  # Squeeze it if monochrome : (1,H,W) -> (H,W) 
        
            filenamei = filename.format(epoch=epoch,i=i)
            
            if len(image.shape) == 2:
                plt.imsave(filenamei, image, cmap='gray_r')
            else:
                plt.imsave(filenamei, image)

    
    
    def on_train_epoch_end(self, trainer, pl_module):
        '''Called at the end of each epoch'''
        
        encoder     = pl_module.encoder
        decoder     = pl_module.decoder

        if self.from_random:
            z      = torch.randn(self.nb_images,self.z_dim).to(device)
            images = decoder(z)
            self.save_images(images.cpu().detach(), self.filename_random, trainer.current_epoch)
            
        if self.from_z:
            z_mean, z_logvar, z  = encoder(self.x.to(device))
            images               = decoder(z)
            self.save_images(images.cpu().detach(), self.filename_z, trainer.current_epoch)


    def get_images(self, epochs=None, from_z=True,from_random=True):
        '''Read and return saved images. epochs is a range'''
        if epochs is None : return
        images_z = []
        images_r = []
        for epoch in list(epochs):
            for i in range(self.nb_images):
                if from_z:
                    f = self.filename_z.format(epoch=epoch,i=i)
                    images_z.append( io.imread(f) )
                if from_random:
                    f = self.filename_random.format(epoch=epoch,i=i)
                    images_r.append( io.imread(f) )
        return images_z, images_r
            
