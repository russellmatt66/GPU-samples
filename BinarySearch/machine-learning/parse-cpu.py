'''
Process `benchmarking-cpu/`, and put raw data into subdirectories. 
Then, calculate runtime statistics from the raw data.
CPU: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz
'''
import pandas as pd
import numpy as np
import sys
import os 

'''
HELPER FUNCTIONS
'''
# Obtain a list of strings representing paths to the datafiles in a sub-sub directory
def getDataFiles(sub_sub_dir: str) -> list[str]:
    # data_files = ['run1.txt']
    print("Getting data files from {}".format(sub_sub_dir))
    data_files = next(os.walk(sub_sub_dir))[2]
    return data_files

# Obtain N from data folder
# `perf_folder` = 'N{#N}_Nx{#Nx}/'
def getN(perf_folder: str) -> int:
    # N = -1
    N = perf_folder.split('_Nx')[0]
    N = int(N.split('N')[1])
    return N

# Obtain Nx from data folder
# `perf_folder` = 'N{#N}_Nx{#Nx}/'
def getNx(perf_folder: str) -> int:
    # Nx = -1
    Nx = perf_folder.split('_Nx')[1]
    Nx = int(Nx.split('/')[0])
    return Nx

# Obtain the number of the run from the name of the datafile
# 'perf_file' = 'run{nrun}.txt'
def getRunNum(perf_file: str) -> int:
    # nrun = -1
    nrun = perf_file.split('.')[0]
    nrun = int(nrun.split('run')[1])
    return nrun

# Obtain runtime from a .txt file representing output from a call to `perf stat`
def getRuntime(perf_file: str) -> float:
    runtime = -1.0
    with open(perf_file, 'r') as data_file:
        for line in data_file:
            rtIdx = line.find("seconds time elapsed") # magic string because of perf stat output
            if (rtIdx != -1):
                runtime = float(line[:rtIdx].strip())
                break
    return runtime

# Parse the immediate sub-directories inside the data heap
# Create a csv where the columns are: [N,Nx,nrun,runtime]
def parseSubDirectory(sub_dir: str) -> pd.DataFrame:
    dict = {} # seed for creating the DataFrame
    features = ['N','Nx','nrun','runtime']
    for feature in features:
        dict[feature] = []
    sub_sub_dirs = next(os.walk(sub_dir))[1] # Read in all sub directories of sub_dir (the sub-sub directories)
    # print(sub_sub_dirs)
    # Loop through the sub directories
    for sub_sub_dir in sub_sub_dirs:     
        sub_sub_dir = sub_sub_dir + "/" # Create path
        N = getN(sub_sub_dir) 
        Nx = getNx(sub_sub_dir) 
        data_files = getDataFiles(sub_dir + sub_sub_dir)
        # print("Data files are {}".format(data_files))
        for data_file in data_files:
            nrun = getRunNum(sub_dir + sub_sub_dir + data_file) # Need to construct path out of strings
            runtime = getRuntime(sub_dir + sub_sub_dir + data_file)
            dict['N'].append(N)
            dict['Nx'].append(Nx)
            dict['nrun'].append(nrun)
            dict['runtime'].append(runtime)
    df = pd.DataFrame(dict)
    return df

'''
MAIN CODE
'''
data_heap = sys.argv[1]
# print(data_heap)

# Get the immediate subdirectories inside data_heap 
particle_sizes = next(os.walk(data_heap))[1]
# print(particle_sizes)

for particle_size in particle_sizes:
    problem_directory = data_heap + particle_size + "/"
    print(problem_directory)
    temp_df = parseSubDirectory(problem_directory)
    temp_df.to_csv(problem_directory + "raw.csv", index=False)