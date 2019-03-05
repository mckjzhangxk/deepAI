import numpy as np
f=lambda x:np.log10(x+1e-14)
print(f(np.random.rand(3,2)))