'''
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
import pandas as pd
import sys
import os

path_to_gpu_stats = sys.argv[1]
path_to_cpu_stats = sys.argv[2]

gpu_df = pd.read_csv(path_to_gpu_stats)
cpu_df = pd.read_csv(path_to_cpu_stats)

speedup_df = gpu_df.sort_values('speedup', ascending=False)
print(speedup_df)

effbw_df = gpu_df.sort_values('eff-bandwidth', ascending=False)
print(effbw_df)

