import torch
from torch import nn
from obj_func.objective import mmsr
from torch import optim 

no_users=10
ob=mmsr(no_users)
loss_function=ob.objective

class mmsr_power(nn.Module):
    def __init__(self,min=0,max=50,no_users=10):
        super().__init__()
        self.power=nn.Parameter(torch.abs(torch.randn(no_users)))
    def forward(self):
        return loss_function(self.power)
power_finder=mmsr_power()
optimizer = torch.optim.SGD(power_finder.parameters(), lr=1e-4)
print(power_finder())

            

