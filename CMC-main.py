import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from c_utils import PolicyNet,QNet
import copy
import gym
import numpy as np
from random import sample
from torch.optim import RMSprop, Adam
import wandb 
import random

wandb.init(project="Actor-Critics Methods for Continous Action spaces",  name="CMC-DDPG")
env = gym.make('MountainCarContinuous-v0')
ndim_states = 2
ndim_actions = 1
minibatch_size = 100
gamma = 0.99
ro = 0.995

D = []
actor_net = PolicyNet(ndim_states,ndim_actions)
critic_net = QNet(ndim_states,ndim_actions)
actor_net_target = copy.deepcopy(actor_net)
critic_net_target = copy.deepcopy(critic_net)
ctr=0

q_optimizer = Adam(critic_net.parameters(), lr=0.001)
pi_optimizer = Adam(actor_net.parameters(), lr=0.001)

run_num = random.randint(10,1000)
rew_arr = []
pi_arr = []
q_arr = []

while(True):
    ctr+=1
    done = False
    s = env.reset()
    #env.set_obs(np.pi,0.0)
    step_ctr=0
    rew=0
    while(done!=True):
        step_ctr+=1

        if ctr>10000:
            a = actor_net(torch.from_numpy(s.astype(np.float32))).detach().numpy() + np.random.normal(0.0, 0.3, ndim_actions)
        else:
            a = env.action_space.sample()

        s_, r, done, info = env.step(a)
        #env.render()
        D.append((s,a,r,s_,done))
        s = s_
        rew+=r
    rew_arr.append(rew)
    wandb.log({"Continuous Mountain Car -v0 Reward Per Episode" :rew},step=ctr)


    if(ctr%25 ==0):
        K=0
        while(K<10):
            K+=1
            B = sample(D,minibatch_size)
            
            #Compute Target
            with torch.no_grad():
                y  = [b[2]+gamma*(1-b[4])*critic_net_target(torch.from_numpy(b[3].astype(np.float32)),actor_net_target(torch.from_numpy(b[3].astype(np.float32)))) for b in B]

            #Update Q-function
            q_optimizer.zero_grad()
            Qsa = [critic_net(torch.from_numpy(b[0].astype(np.float32)),torch.from_numpy(b[1].astype(np.float32))) for b in B ]
            sq_diff = [(Qsa[i]-y[i])**2 for i in range(minibatch_size) ]
            del_Q = torch.stack(sq_diff).sum()/minibatch_size
            wandb.log({"CMC Q-network error":del_Q},step=ctr)
            q_arr.append(del_Q.detach().numpy())
            del_Q.backward()
            q_optimizer.step()

            #Update Policy
            pi_optimizer.zero_grad()
            Q_sao = [critic_net(torch.from_numpy(b[0].astype(np.float32)),actor_net(torch.from_numpy(b[0].astype(np.float32)))) for b in B]
            del_pi = -1*torch.stack(Q_sao).sum()/minibatch_size
            wandb.log({"CMC Pi-network error":del_pi},step=ctr)
            pi_arr.append(del_pi.detach().numpy())
            del_pi.backward()
            pi_optimizer.step()

            #Update target-network
            with torch.no_grad():
                for param,target_param in zip(critic_net.parameters(),critic_net_target.parameters()):
                    target_param.data = ro*target_param.data+(1-ro)*param.data
                for param,target_param in zip(actor_net.parameters(),actor_net_target.parameters()):
                    target_param.data = ro*target_param.data+(1-ro)*param.data


        D=[]  

    if (ctr%5000 == 0 ):
        torch.save({
            'epoch': ctr,
            'ro':ro,
            'actor': actor_net.state_dict(),
            'critic': critic_net.state_dict(),
            'actor_target': actor_net_target.state_dict(),
            'critic_target': critic_net_target.state_dict(),
            'q_optimizer': q_optimizer.state_dict(),
            'pi_optimizer': pi_optimizer.state_dict(),
            }, './checkpoints/Pendulum/'+str(ctr)+'-run'+str(run_num))  

        loss_file = open('./checkpoints/CMC/'+str(run_num)+"reward.txt", "a")
        pi_file = open('./checkpoints/CMC/'+str(run_num)+"pi.txt", "a")
        q_file = open('./checkpoints/CMC/'+str(run_num)+"q.txt", "a")

        loss_file.write(str(rew_arr))  
        pi_file.write(str(pi_arr))
        q_file.write(str(q_arr))

        loss_file.close()
        pi_file.close()
        q_file.close()

        rew_arr = []
        pi_arr = []
        q_arr = []

    
    
