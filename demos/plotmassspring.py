import numpy as np
data = np.loadtxt('output_mass_spring_py.txt', usecols=(0, 1, 2))

import matplotlib.pyplot as plt

plt.plot(data[:,0], data[:,1], label='pos z (Mass A)')
plt.plot(data[:,0], data[:,2], label='vel z (Mass A)')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Mass-Spring (Python) Time Evolution')
plt.legend()
plt.grid()
plt.show()

plt.plot(data[:,1], data[:,2], label='phase plot (z)')
plt.xlabel('position z')
plt.ylabel('velocity z')
plt.title('Mass-Spring (Python) Phase Plot')
plt.legend()
plt.grid()
plt.show()
