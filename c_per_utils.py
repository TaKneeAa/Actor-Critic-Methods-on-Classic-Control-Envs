import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PolicyNet(nn.Module):
    def __init__(self,ndim_states,ndim_actions):
        super().__init__()
        self.nor_input = nn.LayerNorm(ndim_states)
        self.p_fc1 = nn.Linear(ndim_states, 10*ndim_states)
        self.act1 = nn.ReLU()
        self.p_fc2 = nn.Linear(10*ndim_states, 5*ndim_states)
        self.act2 = nn.ReLU()
        self.p_fc3 = nn.Linear(5*ndim_states,ndim_actions)
        self.act3 = nn.Sigmoid()

    def forward(self,x):
        h1 = self.act1(self.p_fc1(self.nor_input(x)))
        h2 = self.act2(self.p_fc2(h1))
        ans = -1.0 + 2.0*self.act3(self.p_fc3(h2))
        return ans


class QNet(nn.Module):
    def __init__(self,ndim_states,ndim_actions):
        super().__init__()
        tot_dim = ndim_states+ndim_actions
        self.nor_input = nn.LayerNorm(tot_dim)
        self.p_fc1 = nn.Linear(tot_dim,10*tot_dim)
        self.act1 = nn.ReLU()
        self.p_fc2 = nn.Linear(10*tot_dim,5*tot_dim)
        self.act2 = nn.ReLU()
        self.p_fc3 = nn.Linear(5*tot_dim,1)


    def forward(self,x,y):
        inp = torch.cat((x,y))
        h1 = self.act1(self.p_fc1(self.nor_input(inp)))
        h2 = self.act2(self.p_fc2(h1))
        ans = self.p_fc3(h2)
        return ans

def perturb_actor(actor_net,param_perturb_std):
    perturbed_actor_net = copy.deepcopy(actor_net)
    for param,per_param in zip(actor_net.parameters(),perturbed_actor_net.parameters()):
        per_param.data = param.data + torch.normal(mean=torch.zeros(param.shape),std=param_perturb_std)
    return perturbed_actor_net

    
