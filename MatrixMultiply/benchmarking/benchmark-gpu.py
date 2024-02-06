import sys
import os
import subprocess
import pandas as pd

# Define a problem space
min_N = 4
max_N = 14 # RTX 2060 limit (~6.0 GB GDDR6)
N = [2**i for i in range(min_N, max_N + 1)]

# Define a configuration space
SM_multipliers_x = [2**i for i in range(7)] # This multiplies the number of device SMs (30 for RTX 2060) to give number of blocks 
SM_multipliers_y = [2**i for i in range(7)]
num_threads_per_blocks_x = [2**i for i in range(5, 10)] 
num_threads_per_blocks_y = [2**i for i in range(5, 10)]

exec_configs = []
for multiplier_x in SM_multipliers_x:
    for multiplier_y in SM_multipliers_y:
        for threads_per_x in num_threads_per_blocks_x:
            for threads_per_y in num_threads_per_blocks_y:
                exec_configs.append((multiplier_x, multiplier_y, threads_per_x, threads_per_y))

# > 12,000 different combinations
# print(exec_configs)
# print(len(exec_configs))
# print(len(exec_configs) * len(N))
 
# Call ./matmul, capture timing output, and store in the appropriate location
