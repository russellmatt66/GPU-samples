'''
Builds, trains, and deploys a DNN to predict timing characteristics for configurations where the timer malfunctioned.

(1) Regression targets are:
    - 'taukern', the execution time of the kernel in milliseconds
    - 'effective-bandwidth', the rate at which the kernel processed data
    - 'speedup', the factor by which the kernel was faster than a single-threaded CPU implementation of the same code 
'''