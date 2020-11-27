from obj_func.objective import mmsr
import torch
from torch import nn

ob=mmsr(10)
loss_function=ob.objective

class mmsr_power(nn.Module):
    def __init__(self,min=0,max=50,no_users=10):
        super().__init__()
        torch.manual_seed(3)
        self.power=nn.Parameter(torch.abs(torch.randn(no_users))+6)
    def forward(self,x):
        return loss_function(x,False)

power_finder=mmsr_power()
no_iters=8000
for i in range(no_iters):

    loss=power_finder(power_finder.power)
    loss.backward(retain_graph=True)

    with torch.no_grad():
        for p in power_finder.parameters():
            p -= p.grad*0.1
            p=torch.clamp(p,min=0,max=10)
    power_finder.zero_grad()   
    if i%400==0:
        print(loss.item())   