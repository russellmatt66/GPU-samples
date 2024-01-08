import pandas as pd

# Analyze the data in a general manner so that this code can be re-used with rtx 2060 data

# Analysis targets:
# (1a) Which run was the absolute fastest, i.e., lowest taukern?
# 
# (1b) Which run achieved the absolute highest effective bandwidth?
# - BW_{eff} = (4 * log2(Nx) * 4 * N) / (taukern * 10^-6) / 10^9
# - The above formula is based on NVIDIA guidelines, and the following specifics:
# - (I) There are N particles to be found, and each particle is represented by 4 bytes of data (float) characterizing its position.
# - (II) For each of these N particles, the binary search algorithm will find them in roughly log2(Nx) iterations.
# - (III) Every iteration, on average, 4 memory operations will be said to occur:
#   -> This number is a compromise. Depending on the exact state of the computation, anywhere from 3 to 5 memory operations will occur.  
# - (IV) To determine the effective bandwidth, divide by the walltime (taukern [ms]), and then divide by a factor of 10^9 to obtain how many GB/s of 
#   data were transferred. 
#
# (2a) Which feature set had the best average performance?
# (2b) Which feature set had the best average bandwidth? 
#
# (3) Which feature set was the most consistent, i.e., lowest variance?
#
# Visualization targets:
# (1) What does the performance landscape look like for a given problem size? 