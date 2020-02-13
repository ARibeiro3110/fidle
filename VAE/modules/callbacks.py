from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

class ImagesCallback(Callback):
    
   
    def __init__(self, filename= 'image-{epoch:03d}-{i:02d}.jpg', z_dim=0, decoder=None, nb_images=5):
        self.filename  = filename
        self.z_dim     = z_dim
        self.decoder   = decoder
        self.nb_images = nb_images
        
        
    def on_epoch_end(self, epoch, logs={}):  
        
        # ---- Get random latent points
        
        z_new   = np.random.normal(size = (self.nb_images,self.z_dim))
        
        # ---- Predict an image
        
        images = self.decoder.predict(np.array(z_new))
        
        # ---- Save images
        
        for i,image in enumerate(images):
            
            # ---- Squeeze it if monochrome : (lx,ly,1) -> (lx,ly) 
            image = image.squeeze()
        
            # ---- Save it

            filename = self.filename.format(epoch=epoch,i=i)
            
            if len(image.shape) == 2:
                plt.imsave(filename, image, cmap='gray_r')
            else:
                plt.imsave(filename, image)
