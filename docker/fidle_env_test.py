#!/usr/bin/env python3
## Some tests to check fidle installation is ok
##

import tensorflow as tf
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

sys.exit(0)
