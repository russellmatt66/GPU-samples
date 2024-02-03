'''
Builds, trains, and deploys a DNN to predict performance characteristics for configurations where the timer malfunctioned.

(1) Regression targets are:
    - 'runtime-avg', the execution time of the kernel in milliseconds
    - 'eff-bandwidth', the rate at which the kernel processed data
    - 'speedup', the factor by which the kernel was faster than a single-threaded CPU implementation of the same code 
    - + the uncertainty measures associated with each target
'''
'''
IMPORT STATEMENTS
'''
import pandas as pd
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
'''
HELPER CODE
'''
# Get the problem sizes for which timer malfunctioned, in order to construct the feature data on which to deploy the DNN
# Copied from `obtain-speedup.py`
def getDirtyN(line: str) -> int:
    line = line.split('.')[0]
    line = line.split('/')[2]
    N = line.split('_Nx')[0]
    N = N.split('N')[1]
    # print(N)
    return int(N)

# Copied from `obtain-speedup.py`
def getDirtyNx(line: str) -> int:
    line = line.split('.')[0]
    line = line.split('/')[2]
    # print(line)
    Nx = line.split('_Nx')[1]
    # print(Nx)
    return int(Nx)

# Create input data for dirty problem sizes
def getDirtyFeatures(path_to_dirty_problems: str, blocks: tuple[int, int], threads_per: tuple[int, int]) -> pd.DataFrame:
    dirty_dict = {}
    features = ['N', 'Nx', 'num_blocks', 'num_threads_per_block']
    for feature in features:
        dirty_dict[feature] = []
    
    num_blocks_min, num_blocks_max = blocks
    num_threads_per_min, num_threads_per_max = threads_per

    # This is probably inefficient, but the execution configuration space is small so I'm mostly concerned with it being correct
    temp_blocks = num_blocks_min
    temp_threads_per = num_threads_per_min
    exec_configs = []
    while (temp_blocks <= num_blocks_max):
        temp_threads_per = num_threads_per_min
        while (temp_threads_per <= num_threads_per_max):
            exec_configs.append((temp_blocks, temp_threads_per))
            temp_threads_per *= 2
        temp_blocks *= 2
    print(exec_configs)

    # Get N and Nx, then add execution configuration data
    with open(path_to_dirty_problems, 'r') as dirty_file:
        for line in dirty_file:
            N = getDirtyN(line)
            Nx = getDirtyNx(line)
            for exec_config in exec_configs:
                dirty_dict['N'].append(N)
                dirty_dict['Nx'].append(Nx)
                dirty_dict['num_blocks'].append(exec_config[0])
                dirty_dict['num_threads_per_block'].append(exec_config[1])

    dirty_df = pd.DataFrame(dirty_dict)
    return dirty_df
'''
MAIN CODE
'''
path_to_gpu_stats = sys.argv[1]
path_to_dirty_problems = sys.argv[2]

gpu_stats = pd.read_csv(path_to_gpu_stats)
print(gpu_stats.head())

input_shape = [4] # Features are [N, Nx, num_blocks, num_threads_per_block]

targets = ['runtime-avg', 'eff-bandwidth', 'speedup','runtime-std','eff-bw-std','speedup-std']
features = ['N', 'Nx', 'num_blocks', 'num_threads_per_block']

X = gpu_stats[features]
y = gpu_stats[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Start with simple, hard-coded example to begin
model = keras.Sequential([
    # Hidden layers
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    # Output layer
    layers.Dense(units=6)
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=256,
    epochs=75,
)

# Add some plotting to visualize loss evolution
# history_df = pd.DataFrame(history.history)
# history_df['loss'].plot()
# plt.show()

# Deploy on the dirty data
# Need bounds for execution configuration in order to construct dirty combinations
(min_blocks_per, max_blocks_per) = (gpu_stats['num_blocks'].min(), gpu_stats['num_blocks'].max())
(min_threads_per, max_threads_per) = (gpu_stats['num_threads_per_block'].min(), gpu_stats['num_threads_per_block'].max())

block_bounds = (min_blocks_per, max_blocks_per)
threads_per_bounds = (min_threads_per, max_threads_per) 
dirty_df = getDirtyFeatures(path_to_dirty_problems, block_bounds, threads_per_bounds)
print(dirty_df.head())
print(dirty_df.shape)

dirty_predictions = model.predict(dirty_df)
print(dirty_predictions.shape)