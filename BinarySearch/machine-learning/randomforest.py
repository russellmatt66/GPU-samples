import pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

'''
# (1) Prediction targets are:
    - 'taukern', the execution time of the kernel in milliseconds
    - 'effective-bandwidth', the rate at which the kernel processed data
    - 'speedup', the factor by which the kernel was faster than a single-threaded CPU implementation of the same code 
    - + error measures for these three values
# (2) Features are the problem size, so N and N_{x}, the number of blocks, and the number of threads per block

Currently, a Random Forest regressor is trained that predicts the three targets, and their errors. 
The MAE, and MRE (Mean Relative Error) of the validation data is computed. 
Next steps:
(1) Play around with the model configuration, and see how accurate it can get. 
(2) Create a DNN, and see what the output is.
'''
'''
HELPER FUNCTIONS
'''
# Read in the problem size of all the 'dirty' problems 
def getDirtySize(dirty_csv: str) -> tuple:
    problem_string = dirty_csv.rsplit('/', 1)[1].split('.')[0].split('_')
    print(problem_string[0])
    N = int(problem_string[0].split('N')[1])
    print(problem_string[1])
    Nx = int(problem_string[1].split('Nx')[1]) 
    return (N,Nx)

'''
MAIN CODE
'''
path_to_gpu_stats = sys.argv[1] # expected to be ./data-analysis/gpu-stats.csv

gpu_stats = pd.read_csv(path_to_gpu_stats) # this will be used for the training, and validation data

dirty_list = sys.argv[2] # expected to be ./*-cleandata/dirty.txt
print(dirty_list)

dirty_problems = []
with open(dirty_list, 'r') as dirty_files:
    for line in dirty_files:
        dirty_problems.append(getDirtySize(line))

print(dirty_problems)

# Implement ML model based on RandomForest architecture in order to predict execution time of a 'dirty' problem
targets = ['runtime-avg', 'eff-bandwidth', 'speedup','runtime-std','eff-bw-std','speedup-std']
features = ['N', 'Nx', 'num_blocks', 'num_threads_per_block']

X = gpu_stats[features]
y = gpu_stats[targets]

print(X)
print(y)

X_train, X_test, y_train, y_val = train_test_split(X, y, random_state=1)

rf_regressor = RandomForestRegressor(random_state=1)
mo_regressor = MultiOutputRegressor(rf_regressor)

mo_regressor.fit(X_train, y_train)

predictions = mo_regressor.predict(X_test)

print(y_val)
print(predictions)
# print(predictions.shape)

# Mean Absolute Error (MAE)
multiple_mae = mean_absolute_error(y_val, predictions, multioutput='raw_values')

for i in range(len(multiple_mae)):
    print("MAE of {} is: {}".format(targets[i], multiple_mae[i]))

# Mean Relative Error (MRE)
relative_errors = np.ndarray(predictions.shape) 

# Scan through the validation data, and calculate relative error between sample and prediction
ip = 0
prediction_indices = [] # Want to have a map between the predictions and the sample they correspond to
for (i, val_row) in y_val.iterrows():
    # print(row)
    j = 0
    for target in val_row.items():
        # print(target)
        # print(target[1])
        val = target[1]
        # type(val)
        relative_errors[ip][j] = abs(predictions[ip][j] - val) / val
        j += 1
    prediction_indices.append(i) # Add associated DataFrame index
    ip += 1

# print(prediction_indices)
# print(len(prediction_indices))
    
multiple_mre = [0.0] * 6

for i in range(len(multiple_mre)):
    multiple_mre[i] = relative_errors[:,i].mean()
    print("MRE of {} is: {}".format(targets[i], multiple_mae[i]))

