import concurrent.futures
import subprocess

# TODO - This needs some more work, but the core idea is for each N, do parallel work for the benchmarking

# TODO - get this from ../benchmarking/benchmark-gpu.py
def parseSTDOUT(stdout: str):
    print("parsing result.stdout")
    return

# Run `matmul`, and collect data into dict
def runMatmul(exec_config: list[tuple], begin: int, end: int, N: int):
    print("Inside 'runMatmul'")
    # N = problem_characteristics[0]
    for i in range(begin, end):
        SM_mult_x = exec_configs[i][0]
        SM_mult_y = exec_configs[i][1]
        num_threads_per_x = exec_configs[i][2]
        num_threads_per_y = exec_configs[i][3]
        result = subprocess.run(['../../build/matmul', str(N), str(SM_mult_x), str(SM_mult_y), str(num_threads_per_x), str(num_threads_per_y)],
            capture_output=True, text=True)
    # Parse the output or perform any other processing as needed
    return result.stdout

def parallelizeMatmulBenchmarking(thread_count: int, N_sizes: list[int], exec_configs: list[tuple]) -> dict:
    print("Inside thread team wrapper")
    
    chunk_size = len(exec_configs) // thread_count
    start_indices = [i * chunk_size for i in range(thread_count)]
    end_indices = start_indices[1:] + [len(exec_configs)]

    print("At thread team call")
    for i in range(len(N_sizes)):
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(executor.map(runMatmul, exec_configs, start_indices, end_indices, N_sizes[i]))

    # Assuming each command corresponds to a key in the dictionary
    # result_dict = dict(zip(commands, results))
    result_dict = dict(results)
    return result_dict

'''
CONFIGURATION
'''
# Define a problem space
min_N = 4
max_N = 14 # RTX 2060 limit (~6.0 GB GDDR6)
N_sizes = [2**i for i in range(min_N, max_N + 1)]

# Define a configuration space
SM_multipliers_x = [2**i for i in range(7)] # This multiplies the number of device SMs (30 for RTX 2060) to give number of blocks 
SM_multipliers_y = [2**i for i in range(7)]
num_threads_per_blocks_x = [2**i for i in range(5, 11)] # [32, 64, 128, 256, 512, 1024] 
num_threads_per_blocks_y = [2**i for i in range(5, 11)]

exec_configs = []
for multiplier_x in SM_multipliers_x:
    for multiplier_y in SM_multipliers_y:
        for threads_per_x in num_threads_per_blocks_x:
            for threads_per_y in num_threads_per_blocks_y:
                exec_configs.append((multiplier_x, multiplier_y, threads_per_x, threads_per_y))
print("configuration specified")
# Example commands
# commands_to_run = ["ls", "echo 'Hello, World!'", "hostname"]

# Number of threads
threads = 8

print("Calling thread team")
result_dict = parallelizeMatmulBenchmarking(threads, N_sizes, exec_configs)
# print(result_dict)
