import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x, y)

axes.set_xlabel("X Label")
axes.set_ylabel("Y Label")

axes.set_title("X vs Y")

plt.show()

