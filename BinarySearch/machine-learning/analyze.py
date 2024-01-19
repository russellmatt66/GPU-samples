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
#   - Obtain this by graphing the average runtime / effective bandwidth against problem space 

clean_datafile = sys.argv[1] # expected to be *-cleandata/cleandata.csv

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
