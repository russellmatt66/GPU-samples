# Overview
Project to implement binary search with CUDA, and then find the configuration with the best performance by performing a parameter sweep of the kernel. 

# Effective Bandwidth Calculation
GPU: GeForce GTX 960 (Maxwell 5.2)
- Theoretical Bandwidth = 112 GB/s
- BW_eff = ((Br + Bw) * avg_iters * 4 * N) / (taukern * 10**-3) / 10**9
    - What the above formula says, is that the volumetric flow of data that the code supported during its operation (a rough measure of performance)
    is equal to the total volume of data that it handled, divided by the runtime ('taukern' is in units of milliseconds), and then converted to GBs.
    - Br + Bw = 3, is taken

# Current Tasks
(1) `machine-learning/`
- Analyze gtx960 kernel benchmarking data
    - `./numiter` simulates the algorithm and obtains an exact value for the number of iterations it takes to find all the particles. 
    - Obtain CPU execution statistics and compare speedup
        - Created bash script for automating the benchmarking + Python wrapper around it
        - Next step is to get all the data, and then write Python for parsing the `perf stat` output into a `.csv` containing runtime 
- Implement an ML model using `sklearn` in order to predict the execution configuration performance for corrupt data.
    - For large data volumes, the output from the CUDA timer library is incoherent, necessitating the usage of models for predicting their performance 

(2) Run project on RTX 2060


# Directory Structure
binarysearch.cu
- Code to benchmark binary search CUDA kernel
- **Appends** data to .csv files
- `$nvcc -o benchmark binarysearch.cu`

binarysearch.c
- CPU code to run binary search on a population of randomly-distributed particles
- `$ gcc binarysearch.c -o cpu-bs`
- `$ ./cpu-bs N Nx`

cpu-bs
- Binary executable for the CPU binary search program

automate-benchmarking.sh
- Shell script that automates the benchmarking of the binary search CUDA kernel
- Need to run `find ./benchmarking-data/ -type f ! -name 'README.md' -exec rm -rf {} +` beforehand to delete everything 

benchmarking-cpu.sh
- Shell script that automates the benchmarking of the CPU binary search program in concert with `benchmarking-cpu.py`

benchmarking-cpu.py
- Python that wraps around `benchmarking-cpu.sh`
- Creates the data folder for every possible problem size 

benchmarking-data/
- Storage for GPU benchmarking data

machine-learning/
- Contains Python code to analyze the performance data and predict the performance as a function of execution configuration 

binarysearch-validate.cu
- Code to validate binary search using linear search
- Linear search kernel will time out before GTX 960 VRAM fills up, only have a single GPU in machine so it's doing both display and compute, therefore watchdog timer

# Benchmarking
**BEFORE RUNNING, MAKE SURE TO DELETE ANY PREVIOUS DATA IN BENCHMARKING-DATA**
`$ ./benchmark-gpu Ni Nx SM_multiplier num_threads_per_block nruns`
- Ni: log2(number of particles)
- Nx: log2(number of gridpoints)
- SM_multiplier: determines the number of blocks in the execution configuration (=SM_multiplier * numberOfSMs)
- num_threads_per_block: the number of threads per block in the execution configuration
- nruns: the number of iterations to run a given configuration for
