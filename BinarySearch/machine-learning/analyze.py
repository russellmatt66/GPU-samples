import pandas as pd
import numpy as np
import glob
import os
import sys

# Analyze the data in a general manner so that this code can be re-used with rtx 2060 data

# Analysis targets:
# (1a) Which configuration was the absolute fastest, i.e., lowest average taukern?
# 
# (1b) Which run achieved the absolute highest effective bandwidth?
# - BW_{eff} = (3 * log2(Nx) * 4 * N) / (taukern * 10^-3) / 10^9
# - The above formula is based on NVIDIA guidelines, and the following specifics:
# - (I) There are N particles to be found, and each particle is represented by 4 bytes of data (float) characterizing its position.
# - (II) For each of these N particles, the binary search algorithm will find them in roughly log2(Nx) iterations.
# - (III) Every iteration, on average, 3 memory operations will be said to occur:
# - (IV) To determine the effective bandwidth, divide by the walltime (taukern [ms]), and then divide by a factor of 10^9 to obtain how many GB/s of 
#   data were transferred. 
#
# (2a) For each given problem size, which feature set (num_blocks, threads_per) had the best average performance?
# - 
# (2b) For each given problem size, which feature set had the best average bandwidth? 
#
# (3) For each given problem size, which feature set was the most consistent, i.e., lowest variance?
#
# (4a) For each given problem size, which execution configuration was the fastest?
#
# (4b) For each given problem size, which execution configuration achieved the highest (average) effective bandwidth?
#
# (5) For each given problem size, what was the average speedup compared to the CPU?
# 
# (6) What are the confidence intervals associated with each of the effective bandwidths? 

# Visualization targets:
# (1) What does the performance landscape look like for a given problem size? 
#   - Obtain this by graphing the average runtime | effective bandwidth against problem space 
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

'''
MAIN CODE
'''
clean_datafile = sys.argv[1] # expected to be *-cleandata/cleandata.csv
cpu_stats = sys.argv[2] # expected to be benchmarking-cpu/stats.csv

clean_df = pd.read_csv(clean_datafile)

# Sort the data by fastest feature set
sorted_df_taukern = clean_df.sort_values('avg_runtime')
print(sorted_df_taukern.head())

# Sort the data by which feature set had the highest effective bandwidth
sorted_df_effbw = clean_df.sort_values('effective_bandwidth', ascending=False)
print(sorted_df_effbw.head())

# Sort the data in ascending order according to execution configuration
sorted_df_execconfig = clean_df.sort_values(['num_blocks', 'num_threads_per_block', 'N', 'Nx'], ascending=[True, True, True, True])
print(sorted_df_execconfig.head())

N_sizes = sorted_df_execconfig['N'].unique()
Nx_sizes = sorted_df_execconfig['Nx'].unique()
print(N_sizes) 
print(Nx_sizes)

problem_sizes = []
# Get dirty problem sizes so that they aren't added to problem_sizes
dirty_problems = []
clean_dir = clean_datafile.split('/')[0] + "/" 
print(clean_dir)
dirty_file = clean_dir + "dirty.txt"

with open(dirty_file, 'r') as dfile:
    for line in dfile:
        dirty_N = getDirtyN(line)
        dirty_Nx = getDirtyNx(line)
        dirty_problems.append((dirty_N, dirty_Nx))
print(dirty_problems)

for N in N_sizes:
    for Nx in Nx_sizes:
        if (N,Nx) not in dirty_problems:
            problem_sizes.append((N,Nx))
print(problem_sizes)