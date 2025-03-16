import numpy as np


print([round(x, 5) for x in np.arange(-1, 1.1, 0.1) if abs(x) > 1e-10])