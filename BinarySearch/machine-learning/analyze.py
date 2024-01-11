import pandas as pd
import glob
import os
import sys

# Analyze the data in a general manner so that this code can be re-used with rtx 2060 data

# Analysis targets:
# (1a) Which configuration was the absolute fastest, i.e., lowest average taukern?
# 
# (1b) Which run achieved the absolute highest effective bandwidth?
# - BW_{eff} = (4 * log2(Nx) * 4 * N) / (taukern * 10^-6) / 10^9
# - The above formula is based on NVIDIA guidelines, and the following specifics:
# - (I) There are N particles to be found, and each particle is represented by 4 bytes of data (float) characterizing its position.
# - (II) For each of these N particles, the binary search algorithm will find them in roughly log2(Nx) iterations.
# - (III) Every iteration, on average, 4 memory operations will be said to occur:
#   -> This number is a compromise. Depending on the exact state of the computation, anywhere from 3 to 5 memory operations will occur.  
# - (IV) To determine the effective bandwidth, divide by the walltime (taukern [ms]), and then divide by a factor of 10^9 to obtain how many GB/s of 
#   data were transferred. 
#
# (2a) Which feature set (num_blocks, threads_per) had the best average performance?
# (2b) Which feature set had the best average bandwidth? 
#
# (3) Which feature set was the most consistent, i.e., lowest variance?
#
# Visualization targets:
# (1) What does the performance landscape look like for a given problem size? 
#   - Obtain this by calculating the effective bandwidth for every variable permutation

def computeStatistics(data_csv: str) -> (int, int, list, list, list, list):
    df = pd.read_csv(data_csv)
    exec_config = [] 
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
    Nx = other_part[2:]

    # Determine the unique execution configurations (should be same for every dataset)
    exec_config.append((df['num_blocks'].unique(),df['num_threads_per_block'].unique()))
    # Compute the average runtime, and variance, for each of the unique execution configurations
    # Compute the (average) effective bandwidth for each of the unique execution configurations
    return N, Nx, exec_config, avg_runtime, variance, eff_bandwidth

# MAIN
kernel_data = sys.argv[1]
# Loop through all the datasets in '*-kerneldata/'
all_files = glob.glob(os.path.join(kernel_data, '**', '*'), recursive=True)

# Filter out directories, leaving only data files
files_only = [file for file in all_files if (os.path.isfile(file) and (file != kernel_data + '/README.md'))]

# Test computeStatistics
(N, Nx, e_c, a_r, v, e_b) = computeStatistics(files_only[0])
print(str(N) + '\n')
print(str(Nx) + '\n')
print(e_c)
print(a_r)
print(v)
print(e_b)