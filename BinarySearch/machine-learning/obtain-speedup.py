import pandas as pd
import numpy as np
import glob
import os
import sys

# Calculate the speedup for each execution configuration

'''
HELPER FUNCTIONS
'''
# Get the problem sizes for which timer malfunctioned, in order to ensure they are excluded from analysis
def getDirtyN(line: str) -> int:
    line = line.split('.')[0]
    line = line.split('/')[2]
    N = line.split('_Nx')[0]
    N = N.split('N')[1]
    # print(N)
    return int(N)

def getDirtyNx(line: str) -> int:
    line = line.split('.')[0]
    line = line.split('/')[2]
    # print(line)
    Nx = line.split('_Nx')[1]
    # print(Nx)
    return int(Nx)

# Wrappers around DataFrame lookups for readability
def getAvgRuntime(gpu_df: pd.DataFrame, N: int, Nx: int, num_blocks: int, num_threads_per_block: int) -> np.float64:
    avg_runtime = gpu_df['avg_runtime'].loc[(gpu_df['N'] == N) 
                             & (gpu_df['Nx'] == Nx) 
                             & (gpu_df['num_blocks'] == num_blocks) 
                             & (gpu_df['num_threads_per_block'] == num_threads_per_block)]
    # print(avg_runtime)
    return avg_runtime.item()

def getRuntimeStd(gpu_df: pd.DataFrame, N: int, Nx: int, num_blocks: int, num_threads_per_block: int) -> np.float64:
    var_runtime = gpu_df['runtime_variance'].loc[(gpu_df['N'] == N) 
                            & (gpu_df['Nx'] == Nx) 
                            & (gpu_df['num_blocks'] == num_blocks) 
                            & (gpu_df['num_threads_per_block'] == num_threads_per_block)]
    # print(var_runtime)
    return np.sqrt(var_runtime.item())

def getEffBw(gpu_df: pd.DataFrame, N: int, Nx: int, num_blocks: int, num_threads_per_block: int) -> np.float64:
    eff_bw = gpu_df['effective_bandwidth'].loc[(gpu_df['N'] == N) 
                            & (gpu_df['Nx'] == Nx) 
                            & (gpu_df['num_blocks'] == num_blocks) 
                            & (gpu_df['num_threads_per_block'] == num_threads_per_block)]
    # print(eff_bw)
    return abs(eff_bw.item())

def getEffBwStd(gpu_df: pd.DataFrame, N: int, Nx: int, num_blocks: int, num_threads_per_block: int) -> np.float64:
    eff_bw_var = gpu_df['effective_bandwidth_variance'].loc[(gpu_df['N'] == N) 
                            & (gpu_df['Nx'] == Nx) 
                            & (gpu_df['num_blocks'] == num_blocks) 
                            & (gpu_df['num_threads_per_block'] == num_threads_per_block)]
    return np.sqrt(eff_bw_var.item())

def getCpuRuntime(cpu_df: pd.DataFrame, N: int, Nx: int) -> np.float64:
    cpu_runtime = cpu_df['runtime-avg'].loc[(cpu_df['N'] == N)
                                            & (cpu_df['Nx'] == Nx)]
    # print(cpu_runtime)
    return cpu_runtime.item()

# Helpers for reading gpu and cpu raw timing data
def getTaukern() -> list[np.float64]:
    return []

def getCPUTiming() -> list[np.float64]:
    return []
'''
MAIN CODE
'''
clean_datafile = sys.argv[1] # expected to be *-cleandata/cleandata.csv
cpu_stats = sys.argv[2] # expected to be benchmarking-cpu/stats.csv

clean_df = pd.read_csv(clean_datafile)

# # Sort the data by fastest feature set
# sorted_df_taukern = clean_df.sort_values('avg_runtime')
# print(sorted_df_taukern.head())

# # Sort the data by which feature set had the highest effective bandwidth
# sorted_df_effbw = clean_df.sort_values('effective_bandwidth', ascending=False)
# print(sorted_df_effbw.head())

# Sort the data in ascending order according to execution configuration
sorted_df_execconfig = clean_df.sort_values(['num_blocks', 'num_threads_per_block', 'N', 'Nx'], ascending=[True, True, True, True])
print(sorted_df_execconfig.head())

# Create a list of the clean problem sizes
problem_sizes = []
N_sizes = sorted_df_execconfig['N'].unique()
Nx_sizes = sorted_df_execconfig['Nx'].unique()
# print(N_sizes) 
# print(Nx_sizes)

# Get dirty problem sizes so that they aren't added to problem_sizes
dirty_problems = []
clean_dir = clean_datafile.split('/')[0] + "/" 
# print(clean_dir)
dirty_file = clean_dir + "dirty.txt"

with open(dirty_file, 'r') as dfile:
    for line in dfile:
        dirty_N = getDirtyN(line)
        dirty_Nx = getDirtyNx(line)
        dirty_problems.append((dirty_N, dirty_Nx))
# print(dirty_problems)

for N in N_sizes:
    for Nx in Nx_sizes:
        if (N,Nx) not in dirty_problems:
            problem_sizes.append((N,Nx))
# print(problem_sizes)

# Create a `gpu-stats.csv`
gpu_dict = {}
features = ['N', 'Nx', 'num_blocks', 'num_threads_per_block', 'runtime-avg', 'runtime-std', 'eff-bandwidth', 'eff-bw-std', 'speedup']
for feature in features:
    gpu_dict[feature] = []

num_blocks = sorted_df_execconfig['num_blocks'].unique()
num_threads_per_block = sorted_df_execconfig['num_threads_per_block'].unique()
print(num_blocks)
print(num_threads_per_block)
exec_configs = []
for blocks in num_blocks:
    for threads in num_threads_per_block:
        exec_configs.append((blocks, threads))

cpu_df = pd.read_csv(cpu_stats)
for problem in problem_sizes:
    N = problem[0]
    Nx = problem[1]
    for exec_config in exec_configs:
        blocks = exec_config[0]
        threads_per = exec_config[1]
        avg_runtime = getAvgRuntime(sorted_df_execconfig, N, Nx, blocks, threads_per)
        std_runtime = getRuntimeStd(sorted_df_execconfig, N, Nx, blocks, threads_per)
        eff_bw = getEffBw(sorted_df_execconfig, N, Nx, blocks, threads_per)
        eff_bw_std = getEffBwStd(sorted_df_execconfig, N, Nx, blocks, threads_per)
        cpu_runtime = getCpuRuntime(cpu_df, N, Nx)
        speedup = cpu_runtime / (avg_runtime*10.0**(-3)) # [avg_runtime] = [ms], [cpu_runtime] = [s] 
        print("(N, Nx, num_blocks, num_threads_per_block) = ({}, {}, {}, {}) gives a speedup of {}".format(N, Nx, blocks, threads_per, speedup))
        gpu_dict['N'].append(N)
        gpu_dict['Nx'].append(Nx)
        gpu_dict['num_blocks'].append(blocks)
        gpu_dict['num_threads_per_block'].append(threads_per)
        gpu_dict['runtime-avg'].append(avg_runtime)
        gpu_dict['runtime-std'].append(std_runtime)
        gpu_dict['eff-bandwidth'].append(eff_bw)
        gpu_dict['eff-bw-std'].append(eff_bw_std)
        gpu_dict['speedup'].append(speedup) 

gpu_df = pd.DataFrame(gpu_dict)
gpu_df.to_csv("./data-analysis/gpu-stats.csv", index=False)        

'''
Compute std of speedup measurements and append to gpu_df
'''
# Why is this not being done previously? 
# Because I didn't think to do it, and it requires going back into the raw timing data.
 
speedup_std = {}
speedup_std['speedup-std'] = []

cpu_timing_data = []
gpu_timing_data = []

for problem_size in problem_sizes:
    N = problem[0]
    Nx = problem[1]
    for exec_config in exec_configs:
        blocks = exec_config[0]
        threads_per = exec_config[1]
        # Go to relevant folder in ./machine-learning/benchmarking-cpu/ 
        # Read timing data from files
        # Go to relevant folder in .machine-learning/*-kerneldata/
        # Read timing data from files
        # Calculate the 