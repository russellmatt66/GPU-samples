''' 
Calculate statistics from raw CPU data
'''
import pandas as pd
import numpy as np
import os
import sys

data_heap = sys.argv[1]

particle_sizes = next(os.walk(data_heap))[1]

raw_dfs = []
for particle_size in particle_sizes:
    problem_directory = data_heap + particle_size + "/"
    print(problem_directory)
    temp_df = pd.read_csv(problem_directory + "raw.csv")
    raw_dfs.append(temp_df)

all_df = pd.concat(raw_dfs)
print(all_df.head())
print(all_df.shape)

master_dict = {}
features = ['N', 'Nx', 'runtime-avg', 'runtime-var']
for feature in features:
    master_dict[feature] = []

N_min = int(np.log2(all_df['N'].min()))
N_max = int(np.log2(all_df['N'].max()))
Nx_min = int(np.log2(all_df['Nx'].min()))
Nx_max = int(np.log2(all_df['Nx'].max()))
print("log2(N) in [{}, {}], log2(Nx) in [{}, {}]".format(N_min, N_max, Nx_min, Nx_max))

N_sizes = [2**i for i in range(N_min, N_max + 1)]
Nx_sizes = [2**j for j in range(Nx_min, Nx_max + 1)]
print(N_sizes)
print(Nx_sizes)

problem_sizes = []
for N in N_sizes:
    for Nx in Nx_sizes:
        problem_sizes.append((N,Nx))
print(len(problem_sizes))

for problem in problem_sizes:
    temp_df = all_df.loc[(all_df['N'] == problem[0]) & (all_df['Nx'] == problem[1])]
    print(temp_df.head())
    print(temp_df.shape)
    avg_runtime = temp_df['runtime'].mean()
    print(avg_runtime)
    var_runtime = temp_df['runtime'].var()
    print(var_runtime)
    master_dict['N'].append(problem[0])
    master_dict['Nx'].append(problem[1])
    master_dict['runtime-avg'].append(avg_runtime)
    master_dict['runtime-var'].append(var_runtime)

master_df = pd.DataFrame(master_dict)
master_df.to_csv(data_heap + "cpu-stats.csv", index=False)