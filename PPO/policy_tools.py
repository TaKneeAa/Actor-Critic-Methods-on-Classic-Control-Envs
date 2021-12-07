import torch
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
import wandb 
import torch
import torch.optim as optim
import torch.nn.functional as F 
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import gym 
from gym.spaces import Dict, Discrete, Box, Tuple


class Random_Policy_Discrete():
    def __init__(self, output_dim):
        self.output_dim = output_dim
    def sample(self):
        return np.random.choice(self.output_dim)

class Random_Policy_Uniform():
    def __init__(self, output_dim, output_range=1):
        self.output_dim = output_dim
        self.output_range = output_range
    def sample(self):
        return self.output_range * torch.rand(self.output_dim)


class Policy_Discrete():
    """
    Implements a two layer neural network policy for discrete action space
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        layer1 = nn.Linear(input_dim, hidden_dim, nn.Tanh())
        layer2 = nn.Linear(hidden_dim, output_dim)
        # softmax_layer = nn.Softmax(dim=output_dim)
        softmax_layer = nn.Softmax()

        self.neural_net = nn.Sequential(layer1, layer2, softmax_layer)

    def sample(self, observation):
        with torch.no_grad():
            action_probs = self.neural_net(observation)
            pred_action = Categorical(probs = action_probs).sample()
            # print(f"Predicted action: {pred_action}")
        return pred_action
    
    def parameters(self):
        return self.neural_net.parameters()
    
    def log_prob(self, observation, action):
        action_probs = self.neural_net(observation)
        prob_fn = Categorical(probs = action_probs)
        log_prob = prob_fn.log_prob(action)
        return log_prob

class Policy_Continuous_v3(nn.Module):
    """
    Implements a Gaussian 3 layer neural network policy for continuous action space. 
    Linear, relu/tanh, linear, relu/tanh , linear
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation = 'tanh'):
        super(Policy_Continuous_v3, self).__init__()
        log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        if activation == 'relu':
            self.action_mean_nn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        else:
            self.action_mean_nn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))

    def sample(self, observation):
        with torch.no_grad():
            action_mean = self.action_mean_nn(observation)
            action_std = torch.exp(self.log_std)
            prob_distn = Normal(action_mean, action_std)
        return prob_distn.sample()

    def log_prob(self, observation, action):
        action_mean = self.action_mean_nn(observation)
        action_std = torch.exp(self.log_std)
        prob_distn = Normal(action_mean, action_std)       
        log_prob_of_each_action_component =  prob_distn.log_prob(action)
        log_prob = log_prob_of_each_action_component.sum(axis=-1)
        return log_prob

class Policy_Continuous_v2(nn.Module):
    """
    Implements a Gaussian two layer neural network policy (Linear, tanh, Linear) for continuous action space. 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation = 'tanh'):
        super(Policy_Continuous_v2, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.mean_projection = nn.Linear(hidden_dim, output_dim)
        log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # Both mean and variance predicting neural nets share a layer. 
        if activation == 'relu':
            self.action_mean_nn = nn.Sequential(self.layer1, nn.ReLU(), self.mean_projection)
        else:
            self.action_mean_nn = nn.Sequential(self.layer1, nn.Tanh(), self.mean_projection)

    
    # def parameters(self):
    #     return list(self.action_mean_nn.parameters()) + [self.log_std]
    
    def sample(self, observation):
        with torch.no_grad():
            action_mean = self.action_mean_nn(observation)
            action_std = torch.exp(self.log_std)
            prob_distn = Normal(action_mean, action_std)
        return prob_distn.sample()

    def log_prob(self, observation, action):
        action_mean = self.action_mean_nn(observation)
        action_std = torch.exp(self.log_std)
        prob_distn = Normal(action_mean, action_std)       
        log_prob_of_each_action_component =  prob_distn.log_prob(action)
        log_prob = log_prob_of_each_action_component.sum(axis=-1)
        return log_prob

class Policy_Continuous():
    """
    Implements a Gaussian two layer neural network policy for continuous action space.
    The standard deviation for any action is also predicted from the observation. The mean and std predicting networks share params.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.mean_projection = nn.Linear(hidden_dim, output_dim)
        self.std_projection = nn.Linear(hidden_dim, output_dim)

        # Both mean and variance predicting neural nets share a layer. 
        # The variance predictor has a final softplus activation as the variance would be positive

        self.action_mean_nn = nn.Sequential(self.layer1, nn.Tanh(), self.mean_projection)
        self.action_std_nn = nn.Sequential(self.layer1, nn.Tanh(), self.std_projection, nn.Softplus())

    def initialise_variables(self):
        pass
    
    def sample(self, observation):
        with torch.no_grad():
            action_mean = self.action_mean_nn(observation)
            action_std = self.action_std_nn(observation)
            prob_distn = Normal(action_mean, action_std)
        return prob_distn.sample()
    
    def parameters(self):
        return list(self.layer1.parameters()) + list(self.mean_projection.parameters()) + list(self.std_projection.parameters())

    def log_prob(self, observation, action):
        action_mean = self.action_mean_nn(observation)
        action_std = self.action_std_nn(observation)
        prob_distn = Normal(action_mean, action_std)       
        log_prob_of_each_action_component =  prob_distn.log_prob(action)
        log_prob = log_prob_of_each_action_component.sum(axis=-1)
        return log_prob