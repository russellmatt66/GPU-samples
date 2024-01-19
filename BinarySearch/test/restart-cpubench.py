# Testing the code for restarting the CPU benchmarking when an error occurs
import sys
import math

benchmarking_path = "./machine-learning/benchmarking-cpu/"
num_runs = 25

# Parsing state
error_string = sys.argv[3]
# print(error_string)

error_string = error_string.split('.txt')[0]
# print(error_string)

nrun_error = error_string.split('run')[1]
# print(nrun_error)

N_error = error_string.split('/N')[1]
# print(N_error)

Nx_error = error_string.split('_Nx')[1]
Nx_error = Nx_error.split('/')[0]
print(Nx_error)

# Picking up from where error occurred
N_max = int(sys.argv[1])
Nx_max = int(sys.argv[2])
Nx_sizes = [2**j for j in range(10, Nx_max + 1)]
# print(N_max)
# print(Nx_max)

Nx_sizes_error = [2**j for j in range(int(math.log2(int(Nx_error))), Nx_max + 1)]
print(Nx_sizes_error)

# Complete work for step that had error
nrun = int(nrun_error)
for Nx in Nx_sizes_error:
    data_folder = benchmarking_path + "N" + str(N_error) + "/" + "N" + str(N_error) + "_Nx" + str(Nx) + "/"
    if (Nx == int(Nx_error)):
        print("Calling `./benchmarking-cpu.sh` with N = {}, Nx = {}, init_run = {}, num_runs = {}, data_folder = {}".format(N_error, Nx, nrun, num_runs, data_folder))
    else: 
        print("Calling `./benchmarking-cpu.sh` with N = {}, Nx = {}, init_run = {}, num_runs = {}, data_folder = {}".format(N_error, Nx, 1, num_runs, data_folder))


# Complete work for rest of run
N_sizes_error = [2**i for i in range(int(math.log2(int(N_error))) + 1, N_max + 1)]
# print(N_sizes_error)
problem_sizes_error = []
for N in N_sizes_error:
    for Nx in Nx_sizes:
        problem_sizes_error.append((N,Nx))
# print(problem_sizes_error)

for problem_size in problem_sizes_error:
    (N, Nx) = problem_size
    data_folder = benchmarking_path + "N" + str(N) + "/" + "N" + str(N) + "_Nx" + str(Nx) + "/"
    print("Calling `./benchmarking-cpu.sh` with N = {}, Nx = {}, init_run = {}, num_runs = {}, data_folder = {}".format(N, Nx, 1, num_runs, data_folder))

        