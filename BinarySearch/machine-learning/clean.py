import pandas as pd
import numpy as np
import sys
import os 
import glob 
import subprocess
'''
Clean the dataset
Here, that means the following:
(1) The dataset is complete, i.e., no missing values, but for sufficiently large problems the CUDA timer malfunctioned, and reported times of either 0 ms 
    or ~10^{-41} ms. For what datasets this occurred must be determined, and they must be removed from the pool. 
(2) Once the malformed datasets are determined, the problems for which they occurred will be recorded. 
    - The ML component of this project will use these problem sizes as the basis for the test set
(3) The remaining clean data will be used in the ML component as the training, and validation datasets.
    - The data analysis component will analyze the clean data in order to compute statistics
'''
def computeStatistics(data_csv: str, num_iter_exe: str) -> (int, int, list, list, list, list):
    df = pd.read_csv(data_csv)
    exec_configs = [] 
    avg_runtime = []
    variance = []
    eff_bandwidth = []
    # Determine the problem size associated with the DataFrame
    # - Use string methods to obtain N, Nx from data_csv
    parts = data_csv.split('.')[0] # get rid of 'csv' ending
    parts = parts.split('/')
    for part in parts: 
        if part.startswith('N'):
            N = int(part[1:])
            break

    other_part = data_csv.split('.')[0] # get rid of 'csv ending
    other_part = other_part.split('_')[1]
    Nx = int(other_part[2:])

    # Determine the unique execution configurations (should be same for every dataset)
    num_blocks = df['num_blocks'].unique()
    num_threads_per_block = df['num_threads_per_block'].unique()
    for blocks in num_blocks:
        for threads_per in num_threads_per_block:
            exec_configs.append((blocks, threads_per)) 
    # exec_configs.append((df['num_blocks'].unique(),df['num_threads_per_block'].unique()))

    # Compute the average runtime, and variance, for each of the unique execution configurations
    for exec_config in exec_configs:
        runtime_vals = df.loc[(df['num_blocks'] == exec_config[0]) & (df['num_threads_per_block'] == exec_config[1]), 'taukern']
        avg_runtime.append(runtime_vals.mean())
        variance.append(runtime_vals.var())

    # print(type(runtime_vals))

    # Compute the (average) effective bandwidth for each of the unique execution configurations
    # 'num_iter_exe' simulates the binary search on the population of particles and calculates the number of iterations required to find all of them
    result = subprocess.check_output([num_iter_exe, str(Nx), str(N)], text=True)
    avg_iter = float(result.strip())
    print("{} iterations needed on average for binary search to find a particle for {} gridpoints, and {} particles".format(avg_iter, Nx, N))
    for runtime in avg_runtime:
        eff_bw = (3.0 * avg_iter * 4.0 * N) / (runtime * 10**-3) / 10**9 # runtime is in milliseconds
        eff_bandwidth.append(eff_bw)

    return N, Nx, exec_configs, avg_runtime, variance, eff_bandwidth

# 
def isDirty(data_csv: str) -> bool:
    print(data_csv)
    df = pd.read_csv(data_csv)
    threshold = 1.0e-9
    return df['taukern'].min() < threshold # malfunctioned data is either =0.0, or =1.0e-41 for this feature

# Program code
kernel_data = sys.argv[1] # Should check that '*-kerneldata/' is taken as input, but who tf is being malicious with this
print("kernel_data = {}".format(kernel_data))

device_id = kernel_data.split('-')[0]
print("device_id = {}".format(device_id))

# Create output folder, '*-cleandata/', for the clean datasets, and list of malformed data 
clean_dir = device_id + '-cleandata/'
print("output location = {}".format(clean_dir))

# os.mkdir(clean_dir) # Add some exception handling to this
try:
    os.mkdir(clean_dir)
    print(f"Directory '{clean_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{clean_dir}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

# Loop through all the datasets in '*-kerneldata/'
all_files = glob.glob(os.path.join(kernel_data, '**', '*'), recursive=True)

# Filter out directories, leaving only files
files_only = [file for file in all_files if (os.path.isfile(file) and (file != kernel_data + 'README.md'))]

malformed_list = clean_dir + 'dirty.txt'
with open(malformed_list, 'w') as dirty_data:
    for file in files_only:
        if isDirty(file):
            dirty_data.write(file + '\n') # Want to record the problem size for which malfunction occurs

# MAIN
# kernel_data = sys.argv[1]
# device_id = kernel_data.split('-')[0]

# Loop through all the datasets in '*-kerneldata/'
# all_files = glob.glob(os.path.join(kernel_data, '**', '*'), recursive=True)

# Filter out directories, leaving only clean data files
dirty_files = []
with open(device_id + '-cleandata/dirty.txt', 'r') as dirty:
    for dirty_file in dirty:
        dirty_files.append(dirty_file.split('\n')[0])

print(dirty_files)

cleanfiles_only = [file for file in all_files if (os.path.isfile(file) and (file != kernel_data + 'README.md')) and (file not in dirty_files)]

stat_df = pd.DataFrame(columns=['N', 'Nx', 'num_blocks', 'num_threads_per_block', 'avg_runtime', 'runtime_variance', 'effective_bandwidth'])

# 'num_iter_exe' simulates the binary search on the population of particles and calculates the number of iterations required to find all of them
num_iter_exe_string = './numiter'

# Test computeStatistics
(N, Nx, e_c, a_r, v, e_b) = computeStatistics(cleanfiles_only[0], num_iter_exe_string)
print(str(N) + '\n')
print(str(Nx) + '\n')
print(e_c)
print(a_r)
print(v)
print(e_b)

# Add computeStatistics results to statistics DataFrame
# The below is slow
for data_file in cleanfiles_only:
    (N, Nx, e_c, a_r, v, e_b) = computeStatistics(data_file, num_iter_exe_string)
    for idx in range(len(e_c)): # number of datapoints based on number of unique execution configurations
        temp_df = pd.DataFrame([[N,Nx,e_c[idx][0],e_c[idx][1],a_r[idx],v[idx],e_b[idx]]],columns = ['N', 'Nx', 'num_blocks', 'num_threads_per_block', 'avg_runtime', 'runtime_variance', 'effective_bandwidth'])
        # stat_df = pd.concat([temp_df, stat_df])
        stat_df = pd.concat([temp_df, stat_df.loc[:, stat_df.notna().any()]], ignore_index=True)
stat_df.to_csv(device_id + '-cleandata/cleandata.csv', index=False)