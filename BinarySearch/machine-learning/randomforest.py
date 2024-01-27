import pandas as pd
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

'''
# (1) Prediction targets are:
    - 'taukern', the execution time of the kernel in milliseconds
    - 'effective-bandwidth', the rate at which the kernel processed data
    - 'speedup', the factor by which the kernel was faster than a single-threaded CPU implementation of the same code 
# (2) Features are the problem size, so N and N_{x}, the number of blocks, and the number of threads per block
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
targets = ['runtime-avg', 'eff-bandwidth', 'speedup']
features = ['N', 'Nx', 'num_blocks', 'num_threads_per_block','runtime-std','eff-bw-std','speedup-std']