import pandas as pd

# Clean the dataset
# Here, that means the following:
# (1) The dataset is complete, i.e., no missing values, but for sufficiently large problems the CUDA timer malfunctioned, and reported times of either 0 ms 
# or ~10^{-41} ms. For what datasets this occurred must be determined, and they must be removed from the pool. 
# (2) Once the malformed datasets are determined, the problems for which they occurred will be recorded. 
# - The ML component of this project will use this list as the basis for the test set
# (3) The remaining clean data will be used in the ML component as the training, and validation datasets.  

# Questions:
# (1) Should this module implement functions to clean kerneldata from a run on a device, or should it implement functionality to create a directory 
#   where the clean data, and list of malformed data, is stored so that it can be read in by analyze.py and randomforest.py?