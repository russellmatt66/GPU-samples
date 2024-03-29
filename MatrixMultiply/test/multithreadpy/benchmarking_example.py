import concurrent.futures
import subprocess
import pandas as pd
import os

'''
THINK IT'S DONE
'''
# Obtain `numberOfSMs`, `device_runtime`, and `host_runtime` from stdout 
def parseSTDOUT(stdout: str) -> dict:
    parse_dict = {}
    # TODO - Obtain `numberOfSMs`, `device_runtime`, and `host_runtime` from stdout
    newlinesplit = stdout.split('\n')
    # print(newlinesplit) 
    # YEP, this is hard-coded with some magic numbers that depend on the output of `../build/matmul`
    # It has to be
    line_numberOfSMs = newlinesplit[1]
    line_CUDAruntime = newlinesplit[2]
    line_CPUMTruntime = newlinesplit[3]
    numberOfSMs = int(line_numberOfSMs.split('=')[1])
    device_runtime = float(line_CUDAruntime.split('=')[1].split('ms')[0])
    host_runtime = float(line_CPUMTruntime.split('=')[1].split('us')[0])
    parse_dict['numberOfSMs'] = numberOfSMs
    parse_dict['device_runtime'] = device_runtime
    parse_dict['host_runtime'] = host_runtime # Currently in [us]
    return parse_dict

def initializeDataDict(data_dict: dict) -> dict:
    features = ['num_run', 'N', 'num_blocks_x', 'num_blocks_y', 'num_threads_per_x', 'num_threads_per_y', 'device_runtime [ms]', 'host_runtime [ms]']
    for feature in features:
        data_dict[feature] = []
    return data_dict

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
        result = subprocess.run(['../../build/matmul', str(N), str(SM_mult_x), str(SM_mult_y), str(num_threads_per_x), str(num_threads_per_y)],
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
        data_dict['host_runtime [ms]'].append(parse_dict['host_runtime'] * 10**-3) # converting [us] to [ms]
    return data_dict

def processN(N: int, exec_configs: list[tuple], nruns: int, thread_count: int, dir_name: str) -> pd.DataFrame:
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        # Use list comprehension to create a list of futures
        futures = [executor.submit(runMatmul, N, config, nruns) for config in exec_configs]

        # Wait for all tasks to complete using as_completed
        raw_dict = {}
        raw_dict = initializeDataDict(raw_dict)
        raw_df = pd.DataFrame(raw_dict)
        for future in concurrent.futures.as_completed(futures):
            # Get, and merge, the benchmarking data of all the workers
            worker_df = pd.DataFrame(future.result())
            # print(worker_df)
            raw_df = pd.concat([raw_df, worker_df], ignore_index=True)
            # print(result)
        raw_df.to_csv(dir_name + 'raw.csv', index=False)

'''
CONFIGURATION
'''
# Define a problem space
min_N = 4
max_N = 5 # RTX 2060 limit (~6.0 GB GDDR6)
N_sizes = [2**i for i in range(min_N, max_N + 1)]

# Define a configuration space
SM_multipliers_x = [2**i for i in range(6)] # This multiplies the number of device SMs (30 for RTX 2060) to give number of blocks 
SM_multipliers_y = [2**i for i in range(6)]
num_threads_per_blocks_x = [2**i for i in range(5, 11)] # [32, 64, 128, 256, 512, 1024] 
num_threads_per_blocks_y = [2**i for i in range(5, 11)]

exec_configs = []
for multiplier_x in SM_multipliers_x:
    for multiplier_y in SM_multipliers_y:
        for threads_per_x in num_threads_per_blocks_x:
            for threads_per_y in num_threads_per_blocks_y:
                exec_configs.append((multiplier_x, multiplier_y, threads_per_x, threads_per_y))
print("configuration specified")

# Create directories
def makeDirectory(data_location: str, N: int) -> str:
    dir_name = data_location + "N" + str(N) + '/'
    try: 
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")
    return dir_name

data_location = './'

dir_names = []
for N in N_sizes:
    dir_names.append(makeDirectory(data_location, N))

# Configure thread team, and number of runs
threads = 8
num_runs = 1

print("Calling thread team")
for N in N_sizes: 
    dir_name = data_location + "N" + str(N) + '/'
    processN(N, exec_configs, num_runs, threads, dir_name)

