import numpy as np 
import torch
class mmsr:
    def __init__(self,no_users):
        torch.manual_seed(4)

        self.secrecy_rate=torch.zeros(no_users) 
        self.no_users=no_users
        
        self.user_gain=self.__gain_init()
        self.eavesdropeer_gain=self.__gain_init(user=False)

    def objective(self,x,no_grad=False):

        for i in range(self.no_users):
            user_rate=self.__user_rate(i,x)
            eavesdropper_rate=self.__eavesdropper_rate(i,x)
            secrecy=user_rate-eavesdropper_rate
            self.secrecy_rate[i]= -1*secrecy
            # print(self.secrecy_rate)
        # print(np.ndarray.max(self.secrecy_rate))  
        if no_grad:  
            print('hi')
            return torch.max(self.secrecy_rate).item()
        return torch.max(self.secrecy_rate)   

    def __user_rate(self,i,x):
        # i is ith user. 
        throughput=torch.sum(self.user_gain[:,i]*x)-(self.user_gain[i,i]*x[i])
        
        rate= torch.log(1+ ((self.user_gain[i,i]*x[i])/throughput))
        return rate
    def __eavesdropper_rate(self,i,x):

        throughput=torch.sum(self.eavesdropeer_gain[0]*x) - (self.eavesdropeer_gain[0][i]*x[i])
        rate= torch.log(1+ ((self.eavesdropeer_gain[0][i]*x[i])/throughput))
        return rate

    def __gain_init(self,user=True):
        if user:
            real=torch.randn(self.no_users,self.no_users)
            imaginary=torch.randn(self.no_users,self.no_users)
        else:
            real=torch.randn(1,self.no_users)
            imaginary=torch.randn(1,self.no_users)

        combined=torch.stack((real,imaginary),dim=2)
        complex_iid=torch.view_as_complex(combined)/torch.sqrt(torch.tensor(2.)) 
        
        return torch.abs(complex_iid)
    





