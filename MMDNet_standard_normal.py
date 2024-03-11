"""
RBF and MMDLoss classes are copied from https://github.com/yiftachbeer/mmd_loss_pytorch with some changes
"""

import numpy as np
import torch
from torch import nn, optim
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm 

class RBF(nn.Module):

    def __init__(self, bandwidth=None): #bandwidth = sigma
        super().__init__()
        self.bandwidth_multipliers = torch.tensor([0.5, 1, 5, 10, 20, 40], dtype = torch.float32)
        self.bandwidth = bandwidth

    def get_bandwidth(self): #if we want to scale the bandwiths
        if self.bandwidth is None:
            return 1
        return self.bandwidth

    def forward(self, X): #calculates the RBF kernel between each column in X
        L2_distances = torch.cdist(X, X) ** 2 
        return torch.exp(-0.5 * L2_distances[None, ...] / ((self.get_bandwidth() * self.bandwidth_multipliers)**2)[:, None, None]).sum(dim=0) - 6*torch.eye(len(X), dtype=torch.float32)
        #dim 0 is the bandwidth dimension : the function returns sum of kernels with different bandwiths defined by self.bandwith_multipliers
        #returns 0 on the diagonal line : dont care for kernel between data point and itself

class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y): #calculates the unbiased empirical MMD between X and Y
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        Y_size = Y.shape[0]
        XX = K[:X_size, :X_size].sum() / (X_size**2-X_size)
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].sum() / (Y_size**2-Y_size)
        return XX - 2 * XY + YY

np.random.seed(0)
torch.manual_seed(0)

# TRAINING DATA:

N = 1000 #number of training data : 50,100,1000
y = np.random.normal(0,1,N) #underlying data distribution is standard gaussian
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1) #create tensor data

# DEFINE MODEL:

N_input = 100 #dimension of uniformly sampled input value
class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(N_input, 200)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(200, 100)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(100, 200)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(200, 1)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x
    
model = network()

loss_fn = MMDLoss() 
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

n_epochs = 80 #10, 80, 150
batch_size = 100 #50 for N=50,100, 100 for N=1000

# TRAINING LOOP:

for epoch in range(n_epochs):
    y = y[torch.randperm(len(y))] #shuffle y
    for i in range(0, len(y), batch_size):
        x = np.random.uniform(-0.5,0.5,batch_size*N_input)
        x = torch.tensor(x, dtype=torch.float32).reshape(batch_size, N_input) #model inputs
        y_pred = model(x) #generated data
        ybatch = y[i:i+batch_size] #real data
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad() #reset gradient
        loss.backward() #backpropagation
        optimizer.step() #optimizer iteration
    print(f'Finished epoch {epoch}, latest loss {loss}')

# TEST MODEL:

X = np.random.uniform(-0.5,0.5,500*N_input)
X = torch.tensor(X, dtype=torch.float32).reshape(500, N_input) #500 random model inputs

with torch.no_grad(): #do not update the gradient 
    y_sample = model(X)

y_sample = y_sample.reshape(1,-1).numpy()[0]
print(y_sample[1:10]) #first 10 generated samples

#histogram
plt.hist(y_sample, bins = 40, density=True) 
x_axis = np.arange(-4, 4, 0.001) 
plt.plot(x_axis, norm.pdf(x_axis, 0, 1)) 
plt.ylabel("Density")
plt.show()

#qq plot
sm.qqplot(y_sample, line='45')
plt.show()

print(np.mean(y_sample)) #generated sample mean
print(np.std(y_sample)) #generated sample sd

# Shapiro-Wilk test for normality, null: data follows normal distribution
from scipy import stats
res = stats.shapiro(y_sample)
#print(res.statistic)
print(res.pvalue)

# Kolmogorov-Smirnov test for goodness of fit, compares to standard normal distribution
print(stats.kstest(y_sample, stats.norm.cdf))