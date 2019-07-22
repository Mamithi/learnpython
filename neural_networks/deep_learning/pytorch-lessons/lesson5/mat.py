import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt



x = np.linspace(0, 10, 100)
y = x**2
y = np.sqrt(x)

plt.plot(x, y, 'r')
plt.xlabel("X Label")
plt.ylabel("Y Label")

plt.title("X vs Y")
plt.show()