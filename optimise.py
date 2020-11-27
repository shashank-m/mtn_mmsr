from obj_func.objective import mmsr
import numpy as np
from scipy.optimize import minimize
import torch 
from numdifftools import Jacobian,Hessian

ob=mmsr(10)
torch.manual_seed(2)
x0=torch.abs(torch.randn(10))+8 # initialise power vector.
# print(x0.tolist())

def set_bounds(max_power=10):
    bounds=[]
    for _ in range(10):
        bounds.append((0,max_power))
    return bounds
bounds=set_bounds()

def fun_der(x):
    return Jacobian(lambda x: ob.objective(x))(x).ravel()

def fun_hess(x):
    return Hessian(lambda x: ob.objective(x))(x)
    
sol=minimize(ob.objective,x0,method='L-BFGS-B',bounds=bounds)
print(sol)