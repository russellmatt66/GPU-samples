#! /bin/bash
# Just a simple bash script to automate the benchmarking of the cpu binary search kernel 

N="$1" # number of particles is log2(N)
Nx="$2" # number of gridpoints is log2(Nx)
init_run="$3" # necessary for being able to restart at a given point due to error
num_runs="$4" # number of times to run perf stat
data_folder="$5" # where to send the output to

<<<<<<< HEAD
for ((nrun = $init_run; nrun <= $num_runs; nrun++)); do
    data_file="$(echo "$data_folder run $nrun .txt" | tr -d ' ')"
    echo $data_file
    perf stat -o $data_file ./cpu-bs $N $Nx 
done
=======
# for ((ni = 10; ni <= $max_N; ni++)); do
#     for ((nx = 10; nx <= $max_Nx; nx++)); do
        for ((nrun = 1; nrun <= num_runs; nrun++)); do
            data_file="$(echo "$data_folder run $nrun .txt" | tr -d ' ')"
            echo $data_file
            perf stat -o $data_file ./cpu-bs $N $Nx 
        done
#     done
# done
>>>>>>> 60f70d064ace9bbc2cdc8e6be002174cfa493c73
