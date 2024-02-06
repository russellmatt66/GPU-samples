import matplotlib.pyplot as plt
import numpy as np

N = np.array([2**i for i in range(4,21)])
data_volume = 12.0 * N**2 / (1024.0)**3 # Convert from bytes into GB

print(np.log2(N[np.where(data_volume >= 6.0)]))

plt.loglog(N, data_volume)
plt.loglog(N, 6.0 * np.ones(N.shape), '--', label='RTX 2060 DRAM')
plt.xlabel('N')
plt.ylabel('Data Volume [GB]')
plt.title('Size of Matmul data volume vs. Number of matrix side elements')
plt.legend()

plt.show()