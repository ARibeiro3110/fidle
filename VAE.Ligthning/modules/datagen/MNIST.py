# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 
# ------------------------------------------------------------------
# 2.0 version by Achille Mbogol Touye, sep 2023


import torch

import numpy as np
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from hashlib import blake2b

# ------------------------------------------------------------------
#   A usefull class to manage our MNIST dataset
#   This class allows to manage datasets derived from the original MNIST
# ------------------------------------------------------------------


class MNIST():
    
    version = '0.1'
    
    def __init__(self):
        pass
   
    @classmethod
    def get_data(cls, normalize=True,  scale=1., train_prop=0.8, shuffle=True, seed=None):
        """
        Return original MNIST dataset
        args:
            normalize   : Normalize dataset or not (True)
            scale      : Scale of dataset to use. 1. mean 100% (1.)
            train_prop : Ratio of train/test (0.8)
            shuffle    : Shuffle data if True (True)
            seed       : Random seed value. False mean no seed, None mean using /dev/urandom (None)
        returns:
            x_train,y_train,x_test,y_test
        """

        # ---- Seed
        #
        if seed is not False:
            np.random.seed(seed)
            print(f'Seeded ({seed})')

        # ---- Get data
        #
        train_dataset = datasets.MNIST(root=".data", train=True,  download=True, transform=T.PILToTensor())

        test_dataset  = datasets.MNIST(root=".data", train=False, download=True, transform=T.PILToTensor())
        print('Dataset loaded.')

        # ---- Normalization
        #
        if normalize:
            train_dataset = datasets.MNIST(root=".data", train=True,  download=True, transform=T.ToTensor())
            test_dataset  = datasets.MNIST(root=".data", train=False, download=True, transform=T.ToTensor())

            trainloader   = DataLoader(train_dataset, batch_size=len(train_dataset))
            testloader    = DataLoader(test_dataset,  batch_size=len(test_dataset) )
            
            x_train       = next(iter(trainloader))[0]
            y_train       = next(iter(trainloader))[1]

            x_test        = next(iter(testloader))[0]
            y_test        = next(iter(testloader))[1]

            print('Normalized.')
            
        else:
            trainloader   = DataLoader(train_dataset, batch_size=len(train_dataset))
            testloader    = DataLoader(test_dataset,  batch_size=len(test_dataset) )
            
            x_train       = next(iter(trainloader))[0]
            y_train       = next(iter(trainloader))[1]

            x_test        = next(iter(testloader))[0]
            y_test        = next(iter(testloader))[1] 

            print('Unnormalized.')
        
        # ---- Concatenate
        #
        x_data = torch.cat([x_train, x_test], dim=0)
        y_data = torch.cat([y_train, y_test])
        print('Concatenated.')

        # ---- Shuffle
        #
        if shuffle:
            p              = torch.randperm(len(x_data))
            x_data, y_data = x_data[p], y_data[p]
            print('Shuffled.')     
        
        # ---- Rescale
        #
        n              = int(scale*len(x_data))
        x_data, y_data = x_data[:n], y_data[:n]
        print(f'rescaled ({scale}).') 

        # ---- Split
        #
        n               = int(len(x_data)*train_prop)
        x_train, x_test = x_data[:n], x_data[n:]
        y_train, y_test = y_data[:n], y_data[n:]
        print(f'splited ({train_prop}).') 

        # ---- Hash
        #
        h = blake2b(digest_size=10)
        for a in [x_train,x_test, y_train,y_test]:
            h.update(a.numpy().tobytes())
            
        # ---- About and return
        #
        print('x_train shape is  : ', x_train.shape)
        print('x_test  shape is  : ', x_test.shape)
        print('y_train shape is  : ', y_train.shape)
        print('y_test  shape is  : ', y_test.shape)
        print('Blake2b digest is : ', h.hexdigest())
        return  x_train,y_train, x_test,y_test
                
            
