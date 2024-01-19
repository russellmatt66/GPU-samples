#! /bin/bash
# Just a simple bash script to automate the benchmarking of the cpu binary search kernel 

N="$1" # number of particles is log2(N)
Nx="$2" # number of gridpoints is log2(Nx)
num_runs="$3" # number of times to run perf stat
data_folder="$4" # where to send the output to

# for ((ni = 10; ni <= $max_N; ni++)); do
#     for ((nx = 10; nx <= $max_Nx; nx++)); do
        for ((nrun = 1; nrun <= num_runs; nrun++)); do
            data_file="$(echo "$data_folder run $nrun .txt" | tr -d ' ')"
            echo $data_file
            perf stat -o $data_file ./cpu-bs N Nx 
        done
#     done
# done
