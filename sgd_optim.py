from obj_func.objective import mmsr
import torch
from torch import nn

ob=mmsr(10)
loss_function=ob.objective


no_iters=600
torch.manual_seed(2)

power_init=torch.abs(torch.randn(10))+8 # initial guess of power

power_init.requires_grad=True

for i in range(no_iters):
    loss=ob.objective(power_init,False) # here loss is max(-secrecy rate) which we are minimising.
    loss.backward(retain_graph=True) # find gradient of loss wrt power.

    power_init.data -= power_init.grad*50 # update power in direction of negative gradient.
    power_init.grad.zero_()

    power_init.data=torch.clamp(power_init.data,min=0,max=10) # putting bound on power b/w 0 and 10.
    

    if i%100==0:
        print(f"iteration {i} loss={loss} power={power_init.data}")
