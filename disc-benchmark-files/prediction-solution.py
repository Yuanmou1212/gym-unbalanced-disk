# %%
import numpy as np
import os
import torch.nn as nn
import torch
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

current_dir = os.getcwd()  
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
target_dir = os.path.join(parent_dir, "Train_ANN", "6_11_test_simulation")
model_path = os.path.join(target_dir, "model.pth")

class Network(nn.Module):
    def __init__(self, n_in, n_hidden_nodes):
        super(Network,self).__init__()
        self.lay1 = nn.Linear(n_in,n_hidden_nodes)
        self.lay2 = nn.Linear(n_hidden_nodes,64)
        self.lay3 = nn.Linear(64,32)
        self.lay4 = nn.Linear(32,1)

    def forward(self,x):
        # print(x.shape)
        x1 = torch.relu(self.lay1(x))
        x2 = torch.relu(self.lay2(x1))
        x3 = torch.relu(self.lay3(x2))
        y = self.lay4(x3)[:,0]
        return y
    
na=6
nb=11
model_pred = Network(na+nb,n_hidden_nodes=64)
model_pred.load_state_dict(torch.load(model_path))
model_pred.eval()

# out = np.load('training-data.npz')
# th_train = out['th'] #th[0],th[1],th[2],th[3],...
# u_train = out['u'] #u[0],u[1],u[2],u[3],...

# data = np.load('test-prediction-submission-file.npz')
data = np.load('test-prediction-submission-file.npz')
upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
# thpred = data['thnow'] #all zeros

def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

device="cuda" if torch.cuda.is_available() else "cpu"
upast_test_tensor = torch.from_numpy(upast_test).to(device=device,dtype=torch.float32)
thpast_test_tensor = torch.from_numpy(thpast_test).to(device=device,dtype=torch.float32)

# %%
# only select the ones that are used in the example
Xtest = torch.cat([upast_test_tensor[:,15-nb:], thpast_test_tensor[:,15-na:]],axis=1)
Xtest = Xtest.to(device=device)
model_pred = model_pred.to(device=device,dtype=torch.float32)
Ypredict = model_pred(Xtest)
Ypredict = Ypredict.reshape(-1,1)
Ypredict = np.concatenate([tensor.detach().cpu().numpy() for tensor in Ypredict])
pd.DataFrame(Ypredict).to_csv((f"prediction_results.csv"))

plt.figure(figsize=(20, 6))
plt.plot(Ypredict, label="prediction")
plt.legend()
pic_name = f"prediction_results.png"
plt.savefig((pic_name))

# %%
assert len(Ypredict)==len(upast_test), 'number of samples changed!!'
np.savez('test-prediction-example-submission-file.npz', upast=upast_test, thpast=thpast_test, thnow=Ypredict)
# %%
