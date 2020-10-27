import numpy as np 
class mmsr:
    def __init__(self,no_users):
        np.random.seed(2)
        self.user_gain= 5*np.abs(np.random.randn(no_users,no_users))
        self.eavesdropeer_gain= np.abs(np.random.randn(no_users))

        self.secrecy_rate=np.zeros(no_users) 
        self.no_users=no_users

    def objective(self,x):

        for i in range(self.no_users):
            user_rate=self.__user_rate(i,x)
            eavesdropper_rate=self.__eavesdropper_rate(i,x)
            secrecy=user_rate-eavesdropper_rate
            self.secrecy_rate[i]=-secrecy
        # print(np.ndarray.max(self.secrecy_rate))   
        return np.ndarray.max(self.secrecy_rate)

    def __user_rate(self,i,x):
        # i is ith user. 
        throughput=np.sum(self.user_gain[:,i]*x)-(self.user_gain[i,i]*x[i])
        
        rate= np.log(1+ ((self.user_gain[i,i]*x[i])/throughput))
        return rate
    def __eavesdropper_rate(self,i,x):

        throughput=np.sum(self.eavesdropeer_gain*x) - (self.eavesdropeer_gain[i]*x[i])
        rate= np.log(1+ ((self.eavesdropeer_gain[i]*x[i])/throughput))
        return rate
