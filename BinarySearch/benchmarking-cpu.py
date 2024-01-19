import sys
import os
import subprocess
import numpy as np

# Create benchmarking directory and data folders
benchmarking_path = "./machine-learning/benchmarking-cpu/"
try:
    os.mkdir(benchmarking_path)
    print(f"Directory '{benchmarking_path}' created successfully.")
except FileExistsError:
    print(f"Directory '{benchmarking_path}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

N_max = int(sys.argv[1]) # largest N benchmarked = 2**N_max
Nx_max = int(sys.argv[2]) # largest Nx benchmarked = 2**Nx_max

N_sizes = [2**i for i in range(10, N_max + 1)]
Nx_sizes = [2**j for j in range(10, Nx_max + 1)]

# print(N_sizes)
# print(Nx_sizes)

problem_sizes = []
for N in N_sizes:
    for Nx in Nx_sizes:
        problem_sizes.append((N,Nx))

# print(problem_sizes)

''' Create a subdirectory for each of the given problem sizes '''
# Create folders for each N
for N in N_sizes:
    data_directory = benchmarking_path + "N" + str(N) + "/"
    try:
        os.mkdir(data_directory)
        print(f"Directory '{data_directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{data_directory}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Create subdirectories for each problem size
for problem_size in problem_sizes:
    (N, Nx) = problem_size
    data_folder = benchmarking_path + "N" + str(N) + "/" + "N" + str(N) + "_Nx" + str(Nx) + "/"
    try:
        os.mkdir(data_folder)
        print(f"Directory '{data_folder}' created successfully.")
    except FileExistsError:
        print(f"Directory '{data_folder}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
# Run bash script that executes `perf stat ./cpu-bs N Nx` an adequate number of times for each of the possible problem sizes 
num_runs = int(sys.argv[3])

for problem_size in problem_sizes:
    (N, Nx) = problem_size
    data_folder = benchmarking_path + "N" + str(N) + "/" + "N" + str(N) + "_Nx" + str(Nx) + "/"
    subprocess.run(["./benchmarking-cpu.sh", str(N), str(Nx), str(num_runs), data_folder])