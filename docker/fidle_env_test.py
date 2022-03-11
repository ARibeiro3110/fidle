#!/usr/bin/env python3
## Some tests to check fidle installation is ok
##

import tensorflow as tf
import torch
import sys, os

# Check data set is found
datasets_dir = os.getenv('FIDLE_DATASETS_DIR', False)
if datasets_dir is False:
   print("FIDLE_DATASETS_DIR not found - Should be /data/fidle_datasets/")
   sys.exit(1) 
print("FIDLE_DATASETS_DIR = ", os.path.expanduser(datasets_dir))

# Check Python version
print("Python version = {}.{}".format(sys.version_info[0], sys.version_info[1]))
# Check tensorflow version
print("Tensorflow version = ", tf.__version__)
# Obsolete command
#print("Tensorflow GPU/CUDA available = ", tf.test.is_gpu_available())
print("Tensorflow GPU/CUDA available = ", "true" if len(tf.config.list_physical_devices('GPU')) else "False")

# Chech Pytorch version
print("Pytorch version = ", torch.__version__)
print("Pytorch GPU/CUDA available = ", torch.cuda.is_available())

sys.exit(0)
