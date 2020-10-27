from objective import mmsr
import numpy as np
from scipy.optimize import minimize

ob=mmsr(10)
np.random.seed(3)
x0=np.abs(np.random.randn(10))
sol=minimize(ob.objective,x0,method='Powell')
print(sol)