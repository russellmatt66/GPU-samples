# Overview
For large N, and Nx, i.e., problems approaching the limits of the GTX 960's capabilities, the CUDA-based timer began to malfunction, producing nonsense. Consequentially, to get a sense of the performance of the code in these regimes, machine learning (ML) models based on the Python scikit-learn (sklearn) library are built, and trained on the valid data.

Beyond this, the goal of this part of the project is also to analyze the datasets for which accurate timing information was obtained.  

# Directory Structure
gtx960-kerneldata/
- Raw, benchmarking data for runs performed on a GeForce GTX 960
- Contains some malformed data, but no missing values
- `taukern` in units of [milliseconds]

gtx960-cleandata/
- Contains .csv containing statistics for all the clean datasets
- Contains .txt listing all the dirty datasets
- `taukern` in units of [milliseconds]

benchmarking-cpu/
- Storage for CPU benchmarking data
- Runtime in units of [seconds]
- CPU: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz

data-analysis/
- Contains results from analyzing the performance data

parse-cpu.py (FIN)
- Parse the output from all the runs of `perf stat`, i.e., the raw CPU benchmarking data in `benchmarking-cpu/`, and distill runtime from it  
- `python3 parse-cpu.py path/to/cpu-bench-data/`

parse-cpu_raw.py (FIN)
- Parse the output from `parse-cpu.py`, and distill execution statistics from it. 
- `python3 parse-cpu_raw.py path/to/cpu-bench-data-raws/`

clean.py (FIN)
- Script that operates on a `*-kerneldata/` folder which contains datasets from a benchmarking run on a GPU
- Produces a directory, `*-clean/`, which contains the clean datasets, and a list of the malformed ones
- TODO - RUN INSTRUCTIONS

analyze.py (WIP)
- Script that operates on a `*-clean/` directory, and computes a number of relevant performance metrics
- TODO - RUN INSTRUCTIONS

obtain-speedup.py (FIN)
- Script that operates on a `*-clean/` directory, and `benchmarking-cpu/cpu-stats.csv` file, and creates `data-analysis/gpu-stats.csv`
- TODO - RUN INSTRUCTIONS

randomforest.py (WIP)
- Code that builds, trains, and deploys a Random Forest model on the gpu-stats dataset from `./data-analysis` directory.
- Purpose of the model is to predict performance, and uncertainty associated with values. 
- `python3 randomforest.py ./data-analysis/gpu-stats.csv ./gtx960-cleandata/dirty.txt`

dnn.py (WIP)
- Code that builds, trains, and deploys a deep neural network to predict timing values based on datasets from a `*-clean/` directory.
- Regression task implemented with `tensorflow` and `keras`.
- `python3 dnn.py ./data-analysis/gpu-stats.csv ./*-cleandata/dirty.txt`

binarytree.h (FIN)
- Library functions for instantiating a binary tree

numiterations.c (FIN)
- Use `binarytree.h` to calculate correct value for 'avg_iters' to put into effective bandwidth formula
    - Assumes uniformly-initialized particles, they were actually randomly-initialized (uniform distribution)
- `$ gcc -std=c99 numiterations.c -o numiter -lm`
- TODO - RUN INSTRUCTIONS

./numiter (FIN)
- Binary created from `numiterations.c`
- Simulates the binary search algorithm and obtains an exact value for the number of iterations it takes to find all the particles. 
- TODO - RUN INSTRUCTIONS

numiterations.cu (FAIL)
- Attempt at writing a CUDA program to accurately determine the number of iterations that it takes, on average, to find a particle using binary search.
    - Given that the population is initialized randomly, according to a uniform distribution.
- Cannot figure out how to write `getCell()`
- `$ nvcc numiterations.cu -o numiter` (don't compile)

project.gv (WIP)
- Graphviz source code for visualizing the structure of the project
- `$ dot -Tpng project.gv -o project.png`

project.png
- Visualizes high-level structure of the project
