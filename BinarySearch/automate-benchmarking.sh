#! /bin/bash
# Just a simple bash script to automate the benchmarking of the BinarySearch kernel

max_N="$1" # number of particles is log2(N)
max_Nx="$2" # number of gridpoints is log2(Nx)
max_sm_multiplier="$3" # number of blocks is $this * numberOfSMs 
max_num_threads_per="$4" # number of threads per block

# Nested for-loops because I'm doing a complete sweep 
for ((nt = 32; nt < $max_num_threads_per; nt *= 2)); do 
    for ((nsm = 1; nsm < $max_sm_multiplier; nsm *= 2)); do
        echo "Benchmarking 2^$ni particles, 2^$nx gridpoints" 
        for ((ni = 10; ni <= $max_N; ni++)); do
            for ((nx = 10; nx <= $max_Nx; nx++)); do
                ./benchmark $ni $nx $nsm $nt 25
            done
        done
        echo "Benchmarking 2^$ni particles, 2^$nx gridpoints complete, moving on." 
    done
done
