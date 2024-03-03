<a name="top"></a>

[<img width="600px" src="fidle/img/title.svg"></img>](#top)

<!-- --------------------------------------------------- -->
<!-- To correctly view this README under Jupyter Lab     -->
<!-- Open the notebook: README.ipynb!                    -->
<!-- --------------------------------------------------- -->

## About Fidle

This repository contains all the documents and links of the **Fidle Training** .   
Fidle (for Formation Introduction au Deep Learning) is a 3-day training session co-organized  
by the 3IA MIAI institute, the CNRS, via the Mission for Transversal and Interdisciplinary  
Initiatives (MITI) and the University of Grenoble Alpes (UGA).  

The objectives of this training are :
 - Understanding the **bases of Deep Learning** neural networks
 - Develop a **first experience** through simple and representative examples
 - Understanding **Tensorflow/Keras** and **Jupyter lab** technologies
 - Apprehend the **academic computing environments** Tier-2 or Tier-1 with powerfull GPU

For more information, see **https://fidle.cnrs.fr** :
- **[Fidle site](https://fidle.cnrs.fr)**
- **[Presentation of the training](https://fidle.cnrs.fr/presentation)**
- **[Detailed program](https://fidle.cnrs.fr/programme)**
- **[Subscribe to the list](https://fidle.cnrs.fr/listeinfo), to stay informed !**
- **[Corrected notebooks](https://fidle.cnrs.fr/done)**
- **[Follow us on our channel :](https://fidle.cnrs.fr/youtube)**\
[<img width="120px" style="vertical-align:middle" src="fidle/img/logo-YouTube.png"></img>](https://fidle.cnrs.fr/youtube)

For more information, you can contact us at :  
[<img width="200px" style="vertical-align:middle" src="fidle/img/00-Mail_contact.svg"></img>](#top)

Current Version : <!-- VERSION_BEGIN -->3.0.9<!-- VERSION_END -->


## Course materials

| | | | |
|:--:|:--:|:--:|:--:|
| **[<img width="50px" src="fidle/img/00-Fidle-pdf.svg"></img><br>Course slides](https://fidle.cnrs.fr/supports)**<br>The course in pdf format<br>| **[<img width="50px" src="fidle/img/00-Notebooks.svg"></img><br>Notebooks](https://fidle.cnrs.fr/notebooks)**<br> &nbsp;&nbsp;&nbsp;&nbsp;Get a Zip or clone this repository &nbsp;&nbsp;&nbsp;&nbsp;<br>| **[<img width="50px" src="fidle/img/00-Datasets-tar.svg"></img><br>Datasets](https://fidle.cnrs.fr/datasets-fidle.tar)**<br>All the needed datasets<br>|**[<img width="50px" src="fidle/img/00-Videos.svg"></img><br>Videos](https://fidle.cnrs.fr/youtube)**<br>&nbsp;&nbsp;&nbsp;&nbsp;Our Youtube channel&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|

Have a look about **[How to get and install](https://fidle.cnrs.fr/installation)** these notebooks and datasets.


## Jupyter notebooks

<!-- TOC_BEGIN -->
<!-- Automatically generated on : 03/03/24 20:38:37 -->

### Linear and logistic regression
- **[LINR1](LinearReg/01-Linear-Regression.ipynb)** - [Linear regression with direct resolution](LinearReg/01-Linear-Regression.ipynb)  
Low-level implementation, using numpy, of a direct resolution for a linear regression
- **[GRAD1](LinearReg/02-Gradient-descent.ipynb)** - [Linear regression with gradient descent](LinearReg/02-Gradient-descent.ipynb)  
Low level implementation of a solution by gradient descent. Basic and stochastic approach.
- **[POLR1](LinearReg/03-Polynomial-Regression.ipynb)** - [Complexity Syndrome](LinearReg/03-Polynomial-Regression.ipynb)  
Illustration of the problem of complexity with the polynomial regression
- **[LOGR1](LinearReg/04-Logistic-Regression.ipynb)** - [Logistic regression](LinearReg/04-Logistic-Regression.ipynb)  
Simple example of logistic regression with a sklearn solution

### Perceptron Model 1957
- **[PER57](Perceptron/01-Simple-Perceptron.ipynb)** - [Perceptron Model 1957](Perceptron/01-Simple-Perceptron.ipynb)  
Example of use of a Perceptron, with sklearn and IRIS dataset of 1936 !

### BHPD regression (DNN), using Keras3/PyTorch
- **[K3BHPD1](BHPD.Keras3/01-DNN-Regression.ipynb)** - [Regression with a Dense Network (DNN)](BHPD.Keras3/01-DNN-Regression.ipynb)  
Simple example of a regression with the dataset Boston Housing Prices Dataset (BHPD)
- **[K3BHPD2](BHPD.Keras3/02-DNN-Regression-Premium.ipynb)** - [Regression with a Dense Network (DNN) - Advanced code](BHPD.Keras3/02-DNN-Regression-Premium.ipynb)  
A more advanced implementation of the precedent example, using Keras3

### BHPD regression (DNN), using PyTorch
- **[PBHPD1](BHPD.PyTorch/01-DNN-Regression_PyTorch.ipynb)** - [Regression with a Dense Network (DNN)](BHPD.PyTorch/01-DNN-Regression_PyTorch.ipynb)  
A Simple regression with a Dense Neural Network (DNN) using Pytorch - BHPD dataset

### Wine Quality prediction (DNN), using Keras3/PyTorch
- **[K3WINE1](Wine.Keras3/01-DNN-Wine-Regression.ipynb)** - [Wine quality prediction with a Dense Network (DNN)](Wine.Keras3/01-DNN-Wine-Regression.ipynb)  
Another example of regression, with a wine quality prediction, using Keras 3 and PyTorch

### Wine Quality prediction (DNN), using PyTorch/Lightning
- **[LWINE1](Wine.Lightning/01-DNN-Wine-Regression-lightning.ipynb)** - [Wine quality prediction with a Dense Network (DNN)](Wine.Lightning/01-DNN-Wine-Regression-lightning.ipynb)  
Another example of regression, with a wine quality prediction, using PyTorch Lightning

### MNIST classification (DNN,CNN), using Keras3/PyTorch
- **[K3MNIST1](MNIST.Keras3/01-DNN-MNIST.ipynb)** - [Simple classification with DNN](MNIST.Keras3/01-DNN-MNIST.ipynb)  
An example of classification using a dense neural network for the famous MNIST dataset
- **[K3MNIST2](MNIST.Keras3/02-CNN-MNIST.ipynb)** - [Simple classification with CNN](MNIST.Keras3/02-CNN-MNIST.ipynb)  
An example of classification using a convolutional neural network for the famous MNIST dataset

### MNIST classification (DNN,CNN), using PyTorch
- **[PMNIST1](MNIST.PyTorch/01-DNN-MNIST_PyTorch.ipynb)** - [Simple classification with DNN](MNIST.PyTorch/01-DNN-MNIST_PyTorch.ipynb)  
Example of classification with a fully connected neural network, using Pytorch

### MNIST classification (DNN,CNN), using PyTorch/Lightning
- **[LMNIST1](MNIST.Lightning/01-DNN-MNIST_Lightning.ipynb)** - [Simple classification with DNN](MNIST.Lightning/01-DNN-MNIST_Lightning.ipynb)  
An example of classification using a dense neural network for the famous MNIST dataset, using PyTorch Lightning
- **[LMNIST2](MNIST.Lightning/02-CNN-MNIST_Lightning.ipynb)** - [Simple classification with CNN](MNIST.Lightning/02-CNN-MNIST_Lightning.ipynb)  
An example of classification using a convolutional neural network for the famous MNIST dataset, using PyTorch Lightning

### Images classification GTSRB with Convolutional Neural Networks (CNN), using Keras3/PyTorch
- **[K3GTSRB1](GTSRB.Keras3/01-Preparation-of-data.ipynb)** - [Dataset analysis and preparation](GTSRB.Keras3/01-Preparation-of-data.ipynb)  
Episode 1 : Analysis of the GTSRB dataset and creation of an enhanced dataset
- **[K3GTSRB2](GTSRB.Keras3/02-First-convolutions.ipynb)** - [First convolutions](GTSRB.Keras3/02-First-convolutions.ipynb)  
Episode 2 : First convolutions and first classification of our traffic signs, using Keras3
- **[K3GTSRB3](GTSRB.Keras3/03-Better-convolutions.ipynb)** - [Training monitoring](GTSRB.Keras3/03-Better-convolutions.ipynb)  
Episode 3 : Monitoring, analysis and check points during a training session, using Keras3
- **[K3GTSRB4](GTSRB.Keras3/04-Keras-cv.ipynb)** - [Hight level example (Keras-cv)](GTSRB.Keras3/04-Keras-cv.ipynb)  
An example of using a pre-trained model with Keras-cv
- **[K3GTSRB10](GTSRB.Keras3/batch_oar.sh)** - [OAR batch script submission](GTSRB.Keras3/batch_oar.sh)  
Bash script for an OAR batch submission of an ipython code
- **[K3GTSRB11](GTSRB.Keras3/batch_slurm.sh)** - [SLURM batch script](GTSRB.Keras3/batch_slurm.sh)  
Bash script for a Slurm batch submission of an ipython code

### Sentiment analysis with word embedding, using Keras3/PyTorch
- **[K3IMDB1](Embedding.Keras3/01-One-hot-encoding.ipynb)** - [Sentiment analysis with hot-one encoding](Embedding.Keras3/01-One-hot-encoding.ipynb)  
A basic example of sentiment analysis with sparse encoding, using a dataset from Internet Movie Database (IMDB), using Keras 3 on PyTorch
- **[K3IMDB2](Embedding.Keras3/02-Keras-embedding.ipynb)** - [Sentiment analysis with text embedding](Embedding.Keras3/02-Keras-embedding.ipynb)  
A very classical example of word embedding with a dataset from Internet Movie Database (IMDB), using Keras 3 on PyTorch
- **[K3IMDB3](Embedding.Keras3/03-Prediction.ipynb)** - [Reload and reuse a saved model](Embedding.Keras3/03-Prediction.ipynb)  
Retrieving a saved model to perform a sentiment analysis (movie review), using Keras 3 and PyTorch
- **[K3IMDB4](Embedding.Keras3/04-Show-vectors.ipynb)** - [Reload embedded vectors](Embedding.Keras3/04-Show-vectors.ipynb)  
Retrieving embedded vectors from our trained model, using Keras 3 and PyTorch
- **[K3IMDB5](Embedding.Keras3/05-LSTM-Keras.ipynb)** - [Sentiment analysis with a RNN network](Embedding.Keras3/05-LSTM-Keras.ipynb)  
Still the same problem, but with a network combining embedding and RNN, using Keras 3 and PyTorch

### Time series with Recurrent Neural Network (RNN), using Keras3/PyTorch
- **[K3LADYB1](RNN.Keras3/01-Ladybug.ipynb)** - [Prediction of a 2D trajectory via RNN](RNN.Keras3/01-Ladybug.ipynb)  
Artificial dataset generation and prediction attempt via a recurrent network, using Keras 3 and PyTorch

### Sentiment analysis with transformer, using PyTorch
- **[TRANS1](Transformers.PyTorch/01-Distilbert.ipynb)** - [IMDB, Sentiment analysis with Transformers ](Transformers.PyTorch/01-Distilbert.ipynb)  
Using a Tranformer to perform a sentiment analysis (IMDB) - Jean Zay version
- **[TRANS2](Transformers.PyTorch/02-distilbert_colab.ipynb)** - [IMDB, Sentiment analysis with Transformers ](Transformers.PyTorch/02-distilbert_colab.ipynb)  
Using a Tranformer to perform a sentiment analysis (IMDB) - Colab version

### Unsupervised learning with an autoencoder neural network (AE), using Keras3
- **[K3AE1](AE.Keras3/01-Prepare-MNIST-dataset.ipynb)** - [Prepare a noisy MNIST dataset](AE.Keras3/01-Prepare-MNIST-dataset.ipynb)  
Episode 1: Preparation of a noisy MNIST dataset
- **[K3AE2](AE.Keras3/02-AE-with-MNIST.ipynb)** - [Building and training an AE denoiser model](AE.Keras3/02-AE-with-MNIST.ipynb)  
Episode 1 : Construction of a denoising autoencoder and training of it with a noisy MNIST dataset.
- **[K3AE3](AE.Keras3/03-AE-with-MNIST-post.ipynb)** - [Playing with our denoiser model](AE.Keras3/03-AE-with-MNIST-post.ipynb)  
Episode 2 : Using the previously trained autoencoder to denoise data
- **[K3AE4](AE.Keras3/04-ExtAE-with-MNIST.ipynb)** - [Denoiser and classifier model](AE.Keras3/04-ExtAE-with-MNIST.ipynb)  
Episode 4 : Construction of a denoiser and classifier model
- **[K3AE5](AE.Keras3/05-ExtAE-with-MNIST.ipynb)** - [Advanced denoiser and classifier model](AE.Keras3/05-ExtAE-with-MNIST.ipynb)  
Episode 5 : Construction of an advanced denoiser and classifier model

### Generative network with Variational Autoencoder (VAE), using Keras3
- **[K3VAE1](VAE.Keras3/01-VAE-with-MNIST-LossLayer.ipynb)** - [First VAE, using functional API (MNIST dataset)](VAE.Keras3/01-VAE-with-MNIST-LossLayer.ipynb)  
Construction and training of a VAE, using functional APPI, with a latent space of small dimension.
- **[K3VAE2](VAE.Keras3/02-VAE-with-MNIST.ipynb)** - [VAE, using a custom model class  (MNIST dataset)](VAE.Keras3/02-VAE-with-MNIST.ipynb)  
Construction and training of a VAE, using model subclass, with a latent space of small dimension.
- **[K3VAE3](VAE.Keras3/03-VAE-with-MNIST-post.ipynb)** - [Analysis of the VAE's latent space of MNIST dataset](VAE.Keras3/03-VAE-with-MNIST-post.ipynb)  
Visualization and analysis of the VAE's latent space of the dataset MNIST

### Generative Adversarial Networks (GANs), using Lightning
- **[PLSHEEP3](DCGAN.Lightning/01-DCGAN-PL.ipynb)** - [A DCGAN to Draw a Sheep, using Pytorch Lightning](DCGAN.Lightning/01-DCGAN-PL.ipynb)  
"Draw me a sheep", revisited with a DCGAN, using Pytorch Lightning

### Diffusion Model (DDPM) using PyTorch
- **[DDPM1](DDPM.PyTorch/01-ddpm.ipynb)** - [Fashion MNIST Generation with DDPM](DDPM.PyTorch/01-ddpm.ipynb)  
Diffusion Model example, to generate Fashion MNIST images.
- **[DDPM2](DDPM.PyTorch/model.py)** - [DDPM Python classes](DDPM.PyTorch/model.py)  
Python classes used by DDMP Example

### Training optimization, using PyTorch
- **[OPT1](Optimization.PyTorch/01-Apprentissages-rapides-et-Optimisations.ipynb)** - [Training setup optimization](Optimization.PyTorch/01-Apprentissages-rapides-et-Optimisations.ipynb)  
The goal of this notebook is to go through a typical deep learning model training

### Deep Reinforcement Learning (DRL), using PyTorch
- **[DRL1](DRL.PyTorch/FIDLE_DQNfromScratch.ipynb)** - [Solving CartPole with DQN](DRL.PyTorch/FIDLE_DQNfromScratch.ipynb)  
Using a a Deep Q-Network to play CartPole - an inverted pendulum problem (PyTorch)
- **[DRL2](DRL.PyTorch/FIDLE_rl_baselines_zoo.ipynb)** - [RL Baselines3 Zoo: Training in Colab](DRL.PyTorch/FIDLE_rl_baselines_zoo.ipynb)  
Demo of Stable baseline3 with Colab

### Miscellaneous things, but very important!
- **[NP1](Misc/00-Numpy.ipynb)** - [A short introduction to Numpy](Misc/00-Numpy.ipynb)  
Numpy is an essential tool for the Scientific Python.
- **[ACTF1](Misc/01-Activation-Functions.ipynb)** - [Activation functions](Misc/01-Activation-Functions.ipynb)  
Some activation functions, with their derivatives.
- **[PANDAS1](Misc/02-Using-pandas.ipynb)** - [Quelques exemples avec Pandas](Misc/02-Using-pandas.ipynb)  
pandas is another essential tool for the Scientific Python.
- **[PYTORCH1](Misc/03-Using-Pytorch.ipynb)** - [Practical Lab : PyTorch](Misc/03-Using-Pytorch.ipynb)  
PyTorch est l'un des principaux framework utilisé dans le Deep Learning
- **[TSB1](Misc/04-Using-Tensorboard.ipynb)** - [Tensorboard with/from Jupyter ](Misc/04-Using-Tensorboard.ipynb)  
4 ways to use Tensorboard from the Jupyter environment
- **[??](Misc/05-RNN.ipynb)** - [??](Misc/05-RNN.ipynb)  
??
<!-- TOC_END -->

## Installation

Have a look about **[How to get and install](https://fidle.cnrs.fr/installation)** these notebooks and datasets.

## Licence

[<img width="100px" src="fidle/img/00-fidle-CC BY-NC-SA.svg"></img>](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
\[en\] Attribution - NonCommercial - ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
\[Fr\] Attribution - Pas d’Utilisation Commerciale - Partage dans les Mêmes Conditions 4.0 International  
See [License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).  
See [Disclaimer](https://creativecommons.org/licenses/by-nc-sa/4.0/#).  


----
[<img width="80px" src="fidle/img/logo-paysage.svg"></img>](#top)
