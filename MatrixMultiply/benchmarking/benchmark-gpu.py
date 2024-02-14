import sys
import os
import subprocess
import pandas as pd

# Define a problem space
min_N = 4
max_N = 14 # RTX 2060 limit (~6.0 GB GDDR6)
N_sizes = [2**i for i in range(min_N, max_N + 1)]

# Define a configuration space
SM_multipliers_x = [2**i for i in range(7)] # This multiplies the number of device SMs (30 for RTX 2060) to give number of blocks 
SM_multipliers_y = [2**i for i in range(7)]
num_threads_per_blocks_x = [2**i for i in range(5, 11)] # [32, 64, 128, 256, 512, 1024] 
num_threads_per_blocks_y = [2**i for i in range(5, 11)]

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

def make_directory(data_location: str, N: int) -> None:
    dir_name = data_location + "N" + str(N)
    try: 
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")
    return

def initialize_data_dict(features: list[str]) -> dict:
    data_dict = {}
    for feature in features:
        data_dict[feature] = []
    return data_dict

# Call ./matmul, capture timing output, and store in the appropriate location
num_runs = int(argv[1])
data_location = '../data/'
features = ['num_run', 'N', 'num_blocks_x', 'num_blocks_y', 'num_threads_per_x', 'num_threads_per_y']

for N in N_sizes:
    data_dict = initialize_data_dict(features)
    make_directory(data_location, N)
    for exec_config in exec_configs:
        SM_mult_x = exec_config[0]
        SM_mult_y = exec_config[1]
        num_threads_per_x = exec_config[2]
        num_threads_per_y = exec_config[3]
        for nrun in range(1, num_runs + 1):
            matmulResult = subprocess.run('../build/matmul', capture_output=True, shell=True, text=True)
            print(matmulResult.stdout)

# benchmarking_df = pd.DataFrame(data_dict)
# benchmarking_df.to_csv('../data/benchmarking-data')