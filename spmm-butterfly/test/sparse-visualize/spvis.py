import numpy as np
import matplotlib.pyplot as plt

Nx = 64

poisson_system = np.diagflat([-2.0] * Nx, 0) + np.diagflat([1.0] * (Nx-1), 1) + np.diagflat([1.0] * (Nx-1), -1)
poisson_system[0][Nx-1] = 1.0
poisson_system[Nx-1][0] = 1.0
print(poisson_system)

plt.matshow(poisson_system)
plt.show()