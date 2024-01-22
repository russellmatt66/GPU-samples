'''
Process `benchmarking-cpu/`, and put raw data into subdirectories. 
Then, calculate runtime statistics from the raw data.
CPU: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz
'''
import pandas as pd
import numpy as np
import sys
import os 
import glob 

'''
HELPER FUNCTIONS
'''
# IMPLEMENT
# Obtain a list of strings representing paths to the datafiles in a sub-sub directory
def getDataFiles(sub_sub_dir: str) -> list[str]:
    data_files = ['run1.txt','run2.txt']
    return data_files

# IMPLEMENT
# Obtain N from datafile
def getN(perf_file: str) -> int:
    N = -1
    return N

# IMPLEMENT
def getNx(perf_file: str) -> int:
    Nx = -1
    return Nx

# IMPLEMENT
# Obtain the number of the run from the name of the datafile
def getRunNum(perf_file: str) -> int:
    nrun = -1
    return nrun

# IMPLEMENT
# Obtain runtime from a .txt file representing output from a call to `perf stat`
def getRuntime(perf_file: str) -> float:
    runtime = -1.0
    return runtime

# IMPLEMENT
# Parse the immediate sub-directories inside the data heap
# Create a csv where the columns are: [N,Nx,nrun,runtime]
def parseSubDirectory(sub_dir: str) -> pd.DataFrame:
    dict = {}
    features = ['N','Nx','nrun','runtime']
    for feature in features:
        dict[feature] = []
    # Read in all sub directories of sub_dir (the sub-sub directories)
    sub_sub_dirs = next(os.walk(sub_dir))[1]
    print(sub_sub_dirs)
    # Loop through the sub directories
    for sub_sub_dir in sub_sub_dirs:     
        # Get a list of datafiles inside sub-sub directory
        sub_sub_dir = sub_sub_dir + "/"  
        # Determine N 
        N = getN(sub_sub_dir) # implement
        # Determine Nx 
        Nx = getNx(sub_sub_dir) # implement
        data_files = getDataFiles(sub_sub_dir)
        print(data_files)
        # Parse all the datafiles: "run{nrun}.txt" inside sub-sub directories
        for data_file in data_files:
            # Calculate nrun
            nrun = getRunNum(data_file)
            # Calculate runtime
            runtime = getRuntime(data_file)
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

# Get the immediate subdirectories inside data_heap 
particle_sizes = next(os.walk(data_heap))[1]
print(particle_sizes)

for particle_size in particle_sizes:
    problem_directory = data_heap + particle_size + "/"
    print(problem_directory)
    temp_df = parseSubDirectory(problem_directory)
    temp_df.to_csv(problem_directory + "raw.csv", index=False)

# Calculate statistics from raw data
# TODO