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
        x1 = torch.relu(self.lay1(x))
        x2 = torch.relu(self.lay2(x1))
        x3 = torch.relu(self.lay3(x2))
        y = self.lay4(x3)[:,0]
        return y

na=6
nb=11
model_sim = Network(na+nb,n_hidden_nodes=64)
model_sim.load_state_dict(torch.load(model_path))
model_sim.eval()

# out = np.load('training-data.npz')
# th_train = out['th'] #th[0],th[1],th[2],th[3],...
# u_train = out['u'] #u[0],u[1],u[2],u[3],...

data = np.load('test-simulation-submission-file.npz')
u_test = data['u']
th_test = data['th'] #only the first 50 values are filled the rest are zeros

def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

na, nb = 6, 11
device="cuda" if torch.cuda.is_available() else "cpu"

u_test_sim_final = [torch.as_tensor(x).to(device=device, dtype=torch.float32) for x in [u_test]][0].squeeze()
th_test_sim_final = [torch.as_tensor(x).to(device=device, dtype=torch.float32) for x in [th_test]][0].squeeze()

def get_NARX_data_final(ulist, f, na, nb,skip=50):
    ulist = ulist.detach().cpu()
    
    # iteratively uses the given f to find the new output.
    upast = u_test_sim_final[skip-nb:skip].detach().cpu()
    ypast = th_test_sim_final[skip-na:skip].detach().cpu()
    ylist = []

    for unow in ulist[skip:]:
        with torch.no_grad():
            ynow = f(upast,ypast).unsqueeze(0)

            #update past arrays
            upast = torch.cat((upast,unow.unsqueeze(0)))[1:]
            ypast = torch.cat((ypast,ynow[0]))[1:]
            
            #save result
            ylist.append(ynow)
            
    return ylist

skip=50
fmodel_sim = lambda u,y: model_sim.forward(torch.cat([u,y]).squeeze().unsqueeze(0))
Ytest_sim_pred_final = get_NARX_data_final(u_test_sim_final, fmodel_sim, na, nb)
output_final = np.concatenate([tensor.cpu().numpy() for tensor in Ytest_sim_pred_final])
first50 = th_test_sim_final[:skip].cpu().numpy().reshape(-1, 1)
output_final = np.concatenate((first50, output_final), axis=0)

pd.DataFrame(output_final).to_csv((f"simulation_results.csv"))
plt.figure(figsize=(20, 6))
plt.plot(output_final, label="simulation")
plt.legend()
pic_name = f"simulation_results.png"
plt.savefig((pic_name))

# %%
assert len(output_final)==len(th_test)
np.savez('test-simulation-example-submission-file.npz', th=output_final, u=u_test)

# %%