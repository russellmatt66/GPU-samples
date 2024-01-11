import pandas as pd
import sys
import os 
import glob 
'''
Clean the dataset
Here, that means the following:
(1) The dataset is complete, i.e., no missing values, but for sufficiently large problems the CUDA timer malfunctioned, and reported times of either 0 ms 
    or ~10^{-41} ms. For what datasets this occurred must be determined, and they must be removed from the pool. 
(2) Once the malformed datasets are determined, the problems for which they occurred will be recorded. 
    - The ML component of this project will use these problem sizes as the basis for the test set
(3) The remaining clean data will be used in the ML component as the training, and validation datasets.
    - The data analysis component will analyze the clean data in order to compute statistics
'''
# 
def isDirty(data_csv: str) -> bool:
    print(data_csv)
    df = pd.read_csv(data_csv)
    threshold = 1.0e-9
    return df['taukern'].min() < threshold # malfunctioned data is either =0.0, or =1.0e-41 for this feature

# Program code
kernel_data = sys.argv[1] # Should check that '*-kerneldata/' is taken as input, but who tf is being malicious with this
print("kernel_data = {}".format(kernel_data))

device_id = kernel_data.split('-')[0]
print("device_id = {}".format(device_id))

# Create output folder, '*-cleandata/', for the clean datasets, and list of malformed data 
clean_dir = device_id + '-cleandata/'
print("output location = {}".format(clean_dir))

# os.mkdir(clean_dir) # Add some exception handling to this
try:
    os.mkdir(clean_dir)
    print(f"Directory '{clean_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{clean_dir}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

# Loop through all the datasets in '*-kerneldata/'
all_files = glob.glob(os.path.join(kernel_data, '**', '*'), recursive=True)

# Filter out directories, leaving only files
files_only = [file for file in all_files if (os.path.isfile(file) and (file != kernel_data + '/README.md'))]

malformed_list = clean_dir + 'dirty.txt'
with open(malformed_list, 'w') as dirty_data:
    for file in files_only:
        if isDirty(file):
            dirty_data.write(file + '\n') # Want to record the problem size for which malfunction occurs