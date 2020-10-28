import numpy as np 
import torch
class mmsr:
    def __init__(self,no_users):
        # np.random.seed(2)
        torch.manual_seed(4)
        self.user_gain= 5*torch.abs(torch.randn(no_users,no_users))
        self.eavesdropeer_gain= torch.abs(torch.randn(no_users))

        self.secrecy_rate=torch.zeros(no_users) 
        self.no_users=no_users

    def objective(self,x):

        for i in range(self.no_users):
            user_rate=self.__user_rate(i,x)
            eavesdropper_rate=self.__eavesdropper_rate(i,x)
            secrecy=user_rate-eavesdropper_rate
            self.secrecy_rate[i]= -1*secrecy
            # print(self.secrecy_rate)
        # print(np.ndarray.max(self.secrecy_rate))   
        return torch.max(self.secrecy_rate).item()

    def __user_rate(self,i,x):
        # i is ith user. 
        throughput=torch.sum(self.user_gain[:,i]*x)-(self.user_gain[i,i]*x[i])
        
        rate= torch.log(1+ ((self.user_gain[i,i]*x[i])/throughput))
        return rate
    def __eavesdropper_rate(self,i,x):

        throughput=torch.sum(self.eavesdropeer_gain*x) - (self.eavesdropeer_gain[i]*x[i])
        rate= torch.log(1+ ((self.eavesdropeer_gain[i]*x[i])/throughput))
        return rate
