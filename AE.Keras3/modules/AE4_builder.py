# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI 2024 - JL. Parouty
# ------------------------------------------------------------------

import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
import keras.layers as layers


# ------------------------------------------------------------------
#   A usefull class to manage our AE model
# ------------------------------------------------------------------


class AE4_builder():
    
    version = '0.1'
    
    def __init__(self,  ae  = { 'latent_dim':16 },
                        cnn = { 'lc1':8, 'lc2':16, 'ld':100 }
                ):
        
        self.ae   = ae
        self.cnn  = cnn
    
    
    
    def create_encoder(self):
        
        latent_dim = self.ae['latent_dim']
                
        inputs    = keras.Input(shape=(28,28,1))
        x         = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
        x         = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x         = layers.Flatten()(x)
        x         = layers.Dense(16,     activation="relu")(x)
        z         = layers.Dense(latent_dim, name='latent')(x)

        encoder = keras.Model(inputs, z, name="encoder")    
        return encoder
        
        
        
    def create_decoder(self):

        latent_dim = self.ae['latent_dim']

        inputs  = keras.Input(shape=(latent_dim,))
        x       = layers.Dense(7 * 7 * 64, activation="relu")(inputs)
        x       = layers.Reshape((7, 7, 64))(x)
        x       = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x       = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name='denoiser')(x)

        decoder = keras.Model(inputs, outputs, name="decoder")
        return decoder
    
        
        
    def create_cnn(self):

        lc1     = self.cnn['lc1']
        lc2     = self.cnn['lc2']
        ld      = self.cnn['ld']

        inputs    = keras.Input(shape=(28, 28, 1))

        x         = keras.layers.Conv2D(lc1, (3,3),  activation='relu')(inputs)
        x         = keras.layers.MaxPooling2D((2,2))(x)
        x         = keras.layers.Dropout(0.2)(x)

        x         = keras.layers.Conv2D(lc2, (3,3), activation='relu')(x)
        x         = keras.layers.MaxPooling2D((2,2))(x)
        x         = keras.layers.Dropout(0.2)(x)

        x         = keras.layers.Flatten()(x)
        x         = keras.layers.Dense(ld, activation='relu')(x)
        x         = keras.layers.Dropout(0.5)(x)

        outputs   = keras.layers.Dense(10, activation='softmax', name='cnn')(x)

        cnn       = keras.Model(inputs, outputs, name='classifier')
        return cnn
    
    
    
    def create_model(self):
        
        # ---- Recover all elementary bricks
        #
        encoder = self.create_encoder()
        decoder = self.create_decoder()
        cnn     = self.create_cnn()
        
        # ---- Build ae
        #
        inputs    = keras.Input(shape=(28, 28, 1))
        latents   = encoder(inputs)
        outputs   = decoder(latents)

        ae = keras.Model(inputs,outputs, name='ae')

        # ---- Assembling final model
        #
        inputs    = keras.Input(shape=(28, 28, 1))

        denoised  = ae(inputs)
        classcat  = cnn(inputs)

        model = keras.Model(inputs, outputs={ 'ae':denoised, 'classifier':classcat})
        
        return model