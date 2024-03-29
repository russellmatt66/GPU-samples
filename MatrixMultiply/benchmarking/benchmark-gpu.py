import sys
import os
import subprocess
import pandas as pd
import concurrent.futures
import time 

'''
CONFIGURATION
'''
# Define a problem space
min_N = 14
max_N = 14 # RTX 2060 limit (~6.0 GB GDDR6)
N_sizes = [2**i for i in range(min_N, max_N + 1)]

# Define a configuration space
SM_multipliers_x = [2**i for i in range(6)] # This multiplies the number of device SMs (30 for RTX 2060) to give number of blocks 
SM_multipliers_y = [2**i for i in range(6)]
num_threads_per_blocks_x = [2**i for i in range(6, 11)] # [64, 128, 256, 512, 1024] 
num_threads_per_blocks_y = [2**i for i in range(6, 11)]

exec_configs = []
for multiplier_x in SM_multipliers_x:
    for multiplier_y in SM_multipliers_y:
        for threads_per_x in num_threads_per_blocks_x:
            for threads_per_y in num_threads_per_blocks_y:
                exec_configs.append((multiplier_x, multiplier_y, threads_per_x, threads_per_y))
print("configuration specified")

# > 12,000 different combinations
# print(exec_configs)
# print(len(exec_configs))
# print(len(exec_configs) * len(N))
'''
HELPER FUNCTIONS
'''
def makeDirectory(data_location: str, N: int) -> str:
    dir_name = data_location + "N" + str(N)
    try: 
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")
    return dir_name

# def initializeDataDict(features: list[str]) -> dict:
#     data_dict = {}
#     for feature in features:
#         data_dict[feature] = []
#     return data_dict
def initializeDataDict(data_dict: dict) -> dict:
    features = ['num_run', 'N', 'num_blocks_x', 'num_blocks_y', 'num_threads_per_x', 'num_threads_per_y', 'device_runtime [ms]']
    for feature in features:
        data_dict[feature] = []
    return data_dict

# TODO - Refactor this for GPU-only benchmarking
# Obtain `numberOfSMs`, `device_runtime`, and `host_runtime` from stdout 
def parseSTDOUT(stdout: str) -> dict:
    parse_dict = {}
    # TODO - Obtain `numberOfSMs`, `device_runtime`, and `host_runtime` from stdout
    newlinesplit = stdout.split('\n')
    # print(newlinesplit) 
    # YEP, this is hard-coded with some magic numbers that depend on the output of `../build/gpu_matmul`
    # It has to be
    line_numberOfSMs = newlinesplit[1]
    line_CUDAruntime = newlinesplit[2]
    numberOfSMs = int(line_numberOfSMs.split('=')[1])
    device_runtime = float(line_CUDAruntime.split('=')[1].split('ms')[0])
    parse_dict['numberOfSMs'] = numberOfSMs
    parse_dict['device_runtime'] = device_runtime
    return parse_dict

# Run `matmul`, and collect data into dict
def runMatmul(N: int, exec_config: tuple, nruns: int) -> dict:
    data_dict = {}
    data_dict = initializeDataDict(data_dict)
    SM_mult_x = exec_config[0]
    SM_mult_y = exec_config[1]
    num_threads_per_x = exec_config[2]
    num_threads_per_y = exec_config[3]
    for i in range(1, nruns + 1):
        print(f"Running N={N}, SM_mult_x={SM_mult_x}, SM_mult_y={SM_mult_y}, num_threads_per_x={num_threads_per_x}, num_threads_per_y={num_threads_per_y}")
        print(f"nrun={i}")
        result = subprocess.run(['../build/gpu_matmul', str(N), str(SM_mult_x), str(SM_mult_y), str(num_threads_per_x), str(num_threads_per_y)],
            capture_output=True, text=True)
        # Parse the output or perform any other processing as needed
        parse_dict = parseSTDOUT(result.stdout)
        data_dict['num_run'].append(i)
        data_dict['N'].append(N)
        data_dict['num_blocks_x'].append(SM_mult_x * parse_dict['numberOfSMs'])
        data_dict['num_blocks_y'].append(SM_mult_y * parse_dict['numberOfSMs'])
        data_dict['num_threads_per_x'].append(num_threads_per_x)
        data_dict['num_threads_per_y'].append(num_threads_per_y)
        data_dict['device_runtime [ms]'].append(parse_dict['device_runtime'])
    return data_dict

'''
BENCHMARKING CODE
'''
# Call ./matmul, capture timing output, and store in the appropriate location
# Configure thread team, and number of runs
threads = 8
num_runs = int(sys.argv[1])
data_location = '../data/'
# features = ['num_run', 'N', 'num_blocks_x', 'num_blocks_y', 'num_threads_per_x', 'num_threads_per_y', 'device_runtime [ms]', 'host_runtime [ms]']

dir_names = [] # Just storing these for good measure 
for N in N_sizes:
    dir_names.append(makeDirectory(data_location, N))

for N in N_sizes:
    data_dict = {}
    data_dict = initializeDataDict(data_dict) # Initialize each value to be an empty list
    dir_name = makeDirectory(data_location, N)
    for exec_config in exec_configs:
        SM_mult_x = exec_config[0]
        SM_mult_y = exec_config[1]
        num_threads_per_x = exec_config[2]
        num_threads_per_y = exec_config[3]
        print(f"Running N={N}, SM_mult_x={SM_mult_x}, SM_mult_y={SM_mult_y}, num_threads_per_x={num_threads_per_x}, num_threads_per_y={num_threads_per_y}")
        # TODO - Implement the multi-threading at this location?
        for nrun in range(1, num_runs + 1):
            print(f"nrun={nrun}")
            matmulResult = subprocess.run(['../build/gpu_matmul', str(N), str(SM_mult_x), str(SM_mult_y), str(num_threads_per_x), str(num_threads_per_y)],
                                          capture_output=True, text=True)
            # print("STDOUT:", matmulResult.stdout)
            # TODO - call parseSTDOUT(matmulResult.stdout), and add data to data_dict, then create a dataframe for the case, and save it to appropriate storage location
            parse_dict = parseSTDOUT(matmulResult.stdout)
            data_dict['num_run'].append(nrun)
            data_dict['N'].append(N)
            data_dict['num_blocks_x'].append(SM_mult_x * parse_dict['numberOfSMs'])
            data_dict['num_blocks_y'].append(SM_mult_y * parse_dict['numberOfSMs'])
            data_dict['num_threads_per_x'].append(num_threads_per_x)
            data_dict['num_threads_per_y'].append(num_threads_per_y)
            data_dict['device_runtime [ms]'].append(parse_dict['device_runtime'])
        print('')
    print('Saving run to ' + dir_name + '/raw.csv\n')
    benchmarking_df = pd.DataFrame(data_dict)    
    benchmarking_df.to_csv(dir_name + '/gpu_raw.csv', index=False)

benchmarking_df = pd.DataFrame(data_dict)
benchmarking_df.to_csv('../data/benchmarking-data')