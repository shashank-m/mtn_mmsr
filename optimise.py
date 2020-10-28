from obj_func.objective import mmsr
import numpy as np
from scipy.optimize import minimize
import torch 

ob=mmsr(10)
torch.manual_seed(2)
x0=torch.abs(torch.randn(10))
sol=minimize(ob.objective,x0,method='Powell')
print(sol)