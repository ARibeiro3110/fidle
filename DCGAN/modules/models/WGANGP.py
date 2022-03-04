# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                            DCGAN Example
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 - S. Arias, E. Maldonado, JL. Parouty
# ------------------------------------------------------------------
# by JL Parouty (dec 2020), based on François Chollet example
#
# Thanks to François Chollet example : https://keras.io/examples

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display,Markdown
import os

class WGANGP(keras.Model):
    '''
    A DCGAN model, built from given generator and discriminator
    '''

    version = '1.0'



    def __init__(self, discriminator=None, generator=None, latent_dim=100, n_critic=3, lambda_gp=10, **kwargs):
        '''
        DCGAN instantiation with a given discriminator and generator
        args :
            discriminator : discriminator model
            generator : generator model
            latent_dim : latent space dimension
        return:
            None
        '''
        super(WGANGP, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator     = generator
        self.latent_dim    = latent_dim
        self.n_critic      = n_critic
        self.lambda_gp     = lambda_gp 
        print(f'Fidle DCGAN is ready :-)  latent dim = {latent_dim}')


        
    def call(self, inputs):
        '''
        Implementation of the model forward pass
        args:
            inputs : vectors from latent space
        return:
            output : Output of the generator
        '''
        outputs = self.generator(inputs)
        return outputs
                


    def compile(self, 
                discriminator_optimizer = keras.optimizers.Adam(), 
                generator_optimizer     = keras.optimizers.Adam()
                ):
        '''
        Compile the model
        args:
            discriminator_optimizer : Discriminator optimizer (Adam)
            generator_optimizer : Generator optimizer (Adam)
            loss_function : Loss function
        '''
        super(WGANGP, self).compile()
        self.discriminator.compile(optimizer=discriminator_optimizer)
        self.generator.compile(optimizer=generator_optimizer)
        
        self.d_optimizer   = discriminator_optimizer
        self.g_optimizer   = generator_optimizer
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")



    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]


    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Create some interpolated image
        epsilon = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = fake_images + epsilon * (real_images - fake_images)

        # Calculate interpolated critics, in gradient tape mode
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # Get the critics for the interpolated image.
            interpolated_critics = self.discriminator(interpolated, training=True)

        # Retrieve gradients for this interpolated critics
        gradients = gp_tape.gradient(interpolated_critics, [interpolated])[0]
        # Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        # Calculate the final gp
        gp = self.lambda_gp * tf.reduce_mean((norm - 1.0) ** 2)
        return gp



    def train_step(self, inputs):
        '''
        Implementation of the training update.
        Receive some real images.
        This will compute loss, get gradients and update weights for generator and discriminator
        Return metrics.
        args:
            real_images : real images
        return:
            d_loss  : discriminator loss
            g_loss  : generator loss
        '''

        # ---- Get the input we need, specified in the .fit()
        #      inputs is tensor or a tuple of tensors
        #
        if isinstance(inputs, tuple):
            real_images = inputs[0]
        else:
            real_images = inputs

        batch_size=tf.shape(real_images)[0]

        # ---- Train the discriminator ----------------------------------------
        # ---------------------------------------------------------------------
        #  d_loss = D(fake) - D(real) + lambda.( ||d D(interpolated)|| - 1 )^2
        #         =        w_loss     + gp    
        #
        for i in range(self.n_critic):
            
            # ---- Forward pass
            #
            # Get some random points in the latent space : z
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Generate fake images with the generator : G(z)
            fake_images = self.generator(random_latent_vectors, training=True)

            # Record operations with the GradientTape.
            with tf.GradientTape() as tape:

                # Get critics for the fake images : D(G(z))
                fake_critics = self.discriminator(fake_images, training=True)
                
                # Get critics for the real images : D(x)
                real_critics = self.discriminator(real_images, training=True)

                # Calculate the wasserstein discriminator loss L = D(fake) - D(real)
                w_loss = tf.reduce_mean(fake_critics) - tf.reduce_mean(real_critics)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)

                # Calculate the full discriminator loss : loss = w_loss + gp
                d_loss = w_loss + gp

            # ---- Backward pass
            #
            # Retrieve gradients from gradient_tape and run one step 
            # of gradient descent to optimize trainable weights
            gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)

            # Update discriminator weights
            self.d_optimizer.apply_gradients( zip(gradients, self.discriminator.trainable_weights) )

        # ---- Train the generator --------------------------------------------
        # ---------------------------------------------------------------------
        # g_loss = -D(fake)
        #
        # ---- Forward pass
        #
        # Get some random points in the latent space : z
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Record operations with the GradientTape.
        with tf.GradientTape() as tape:

            # Generate fake images using the generator
            fake_images = self.generator(random_latent_vectors, training=True)

            # Get critics for fake images
            fake_critics = self.discriminator(fake_images, training=True)

            # Calculate the generator loss
            g_loss = -tf.reduce_mean(fake_critics)

        # ---- Backward pass
        #
        # Retrieve gradients from gradient_tape and run one step 
        # of gradient descent to optimize trainable weights
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)

        # Update generator weights
        self.g_optimizer.apply_gradients( zip(gradients, self.generator.trainable_variables) )

        # ---- Update and return metrics --------------------------------------
        # ---------------------------------------------------------------------
        #
        # return {
        #     "d_loss": d_loss,
        #     "g_loss": g_loss
        # }

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


            
    def save(self,filename):
        '''Save model in 2 part'''
        save_dir             = os.path.dirname(filename)
        filename, _extension = os.path.splitext(filename)
        # ---- Create directory if needed
        os.makedirs(save_dir, mode=0o750, exist_ok=True)
        # ---- Save models
        self.discriminator.save( f'{filename}-discriminator.h5' )
        self.generator.save(     f'{filename}-generator.h5'     )

    
    def reload(self,filename):
        '''Reload a 2 part saved model.
        Note : to train it, you need to .compile() it...'''
        filename, extension = os.path.splitext(filename)
        self.discriminator = keras.models.load_model(f'{filename}-discriminator.h5', compile=False)
        self.generator     = keras.models.load_model(f'{filename}-generator.h5'    , compile=False)
        print('Reloaded.')
                
        
    @classmethod
    def about(cls):
        '''Basic whoami method'''
        display(Markdown('<br>**FIDLE 2022 - DCGAN**'))
        print('Version              :', cls.version)
        print('TensorFlow version   :', tf.__version__)
        print('Keras version        :', tf.keras.__version__)
