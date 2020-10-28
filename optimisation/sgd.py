import torch
from torch import nn
from obj_func.objective import mmsr
from torch import optim 


class modified_mmsr(mmsr):
    def __init__(self,no_users=10):
        super().__init__(no_users)

    def objective(self,x):
        for i in range(self.no_users):
            user_rate=self._mmsr__user_rate(i,x)
            eavesdropper_rate=self._mmsr__eavesdropper_rate(i,x)
            secrecy=user_rate-eavesdropper_rate
            self.secrecy_rate[i]= -1*secrecy

        return torch.max(self.secrecy_rate)    

no_users=10
ob=modified_mmsr(no_users)
loss_function=ob.objective


class mmsr_power(nn.Module):
    def __init__(self,min=0,max=50,no_users=10):
        super().__init__()
        torch.manual_seed(3)
        self.power=nn.Parameter(torch.abs(torch.randn(no_users)))
    def forward(self,x):
        return loss_function(x)

power_finder=mmsr_power()

no_iters=3
for i in range(no_iters):

    loss=power_finder(power_finder.power)
    loss.backward(retain_graph=True)

    print(loss)

    with torch.no_grad():
        for p in power_finder.parameters():
            p -= p.grad*0.1

    power_finder.zero_grad()        
        
