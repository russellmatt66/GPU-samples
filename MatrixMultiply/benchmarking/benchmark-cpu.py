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
min_N = 4
max_N = 14 # RTX 2060 limit (~6.0 GB GDDR6)
N_sizes = [2**i for i in range(min_N, max_N + 1)]

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

def initializeDataDict(data_dict: dict) -> dict:
    features = ['num_run', 'N', 'host_runtime [ms]']
    for feature in features:
        data_dict[feature] = []
    return data_dict

# TODO - Refactor this for CPU benchmarking 
#  Obtain `numberOfSMs`, `device_runtime`, and `host_runtime` from stdout 
def parseSTDOUT(stdout: str) -> dict:
    parse_dict = {}
    # TODO - Obtain `host_runtime` from stdout
    newlinesplit = stdout.split('\n')
    # print(newlinesplit) 
    # Has to be hard-coded with some magic numbers that depend on the output of `../build/cpu_matmul`
    line_CPUMTruntime = newlinesplit[0]
    host_runtime = float(line_CPUMTruntime.split('=')[1].split('us')[0])
    # print(host_runtime)
    parse_dict['host_runtime'] = host_runtime # Currently in [us]
    return parse_dict

# TODO - Refactor this for CPU benchmarking
# Run `cpu_matmul`, and collect data into dict
def runMatmul(N: int, nrun: int) -> dict:
    data_dict = {}
    data_dict = initializeDataDict(data_dict)
    print(f"Running N={N}")
    print(f"nrun={nrun}")
    result = subprocess.run(['../build/cpu_matmul', str(N)],
        capture_output=True, text=True)
    # Parse the output or perform any other processing as needed
    parse_dict = parseSTDOUT(result.stdout)
    data_dict['num_run'].append(nrun)
    data_dict['N'].append(N)
    data_dict['host_runtime [ms]'].append(parse_dict['host_runtime'] * 10**-3) # converting [us] to [ms]
    return data_dict

# TODO - Refactor this for CPU benchmarking
def processN(N: int, nruns: int, thread_count: int, dir_name: str) -> pd.DataFrame:
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        # Use list comprehension to create a list of futures
        futures = [executor.submit(runMatmul, N, nrun) for nrun in range(1, nruns + 1)]

        completed_futures, _ = concurrent.futures.wait(futures)
        
        # Wait for all tasks to complete using as_completed
        raw_dict = {}
        raw_dict = initializeDataDict(raw_dict)
        raw_df = pd.DataFrame(raw_dict)
        # for future in concurrent.futures.as_completed(futures):
        for future in completed_futures:
            # print(future.result())
            # Get, and merge, the benchmarking data of all the workers
            worker_df = pd.DataFrame(future.result())
            # print(worker_df)
            raw_df = pd.concat([raw_df, worker_df], ignore_index=True)
            # print(result)
        raw_df.to_csv(dir_name + 'raw_cpu.csv', index=False)

'''
BENCHMARKING CODE
'''
# Call ./cpu_matmul, capture timing output, and store in the appropriate location
# Configure thread team, and number of runs
threads = 8
num_runs = int(sys.argv[1])
data_location = '../data/'
features = ['num_run', 'N', 'host_runtime [ms]']

dir_names = [] # Just storing these for good measure 
for N in N_sizes:
    dir_names.append(makeDirectory(data_location, N))

print("Calling thread team")
start_time = time.time()
for N in N_sizes: 
    dir_name = data_location + "N" + str(N) + '/'
    processN(N, num_runs, threads, dir_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("Benchmarking took {elapsed_time} seconds")