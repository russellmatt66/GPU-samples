import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Need to come up with a pipeline for handling the large number of data files in a general manner
# Workflow for a new dataset is: clean.py -> analyze.py -> randomforest.py 

# (1) Prediction target is 'taukern', the execution time of the kernel in milliseconds
# (2) Features are the problem size, so N and N_{x}, the number of blocks, and the number of threads per block