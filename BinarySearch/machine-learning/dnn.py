'''
Builds, trains, and deploys a DNN to predict timing characteristics for configurations where the timer malfunctioned.

(1) Regression targets are:
    - 'runtime-avg', the execution time of the kernel in milliseconds
    - 'eff-bandwidth', the rate at which the kernel processed data
    - 'speedup', the factor by which the kernel was faster than a single-threaded CPU implementation of the same code 
    - + the uncertainty measures associated with each target
'''
'''
HELPER CODE
'''
# WIP
'''
MAIN CODE
'''
import pandas as pd
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

path_to_gpu_stats = sys.argv[1]

gpu_stats = pd.read_csv(path_to_gpu_stats)
gpu_stats.head()

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
    layers.Dense(units=1, name='avg-runtime'),
    layers.Dense(units=1, name='eff-bandwidth'),
    layers.Dense(units=1, name='speedup'),
    layers.Dense(units=1, name='std_avg-runtime'),
    layers.Dense(units=1, name='std_eff-bandwidth'),
    layers.Dense(units=1, name='std_speedup')
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=256,
    epochs=100,
)

# Add some plotting 
history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
plt.show()