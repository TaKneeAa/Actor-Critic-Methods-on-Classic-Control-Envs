"""
Implement REINFORCE (without baseline) 
using this Pseudocode: https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""
import torch
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F 
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import gym 
from gym.spaces import Dict, Discrete, Box, Tuple

np.random.seed(1)
torch.manual_seed(1)

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

class Policy_Continuous_v2():
    """
    Implements a Gaussian two layer neural network policy for continuous action space. 

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

class Policy_Continuous():
    """
    Implements a Gaussian two layer neural network policy for continuous action space
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


class Value_Function_Apx(nn.Module):
    """
    Implements a two layer neural network that approximates the Value Function
    """
    def __init__(self, input_dim, hidden_dim):
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim, activation = 'tanh')
        self.layer2 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, observation):
        """
        """
        layer1_output = self.layer1(observation)
        layer2_output = self.layer2(layer1_output)
        return layer2_output

class MDP():
    def __init__(self, environment_name, gamma, hidden_dim, horizon):
        print(f"Initialising environment: {environment_name}")
        self.env = gym.make(environment_name)
        self.env._max_episode_steps = 2000
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.input_dim = self.env.observation_space.shape[0]

        self.display_environment_basics()
        if isinstance(self.env.action_space, Discrete):
            self.policy_is_discrete = True
            output_dim = self.env.action_space.n
            self.policy = Policy_Discrete(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = output_dim)
        else:
            self.policy_is_discrete = False
            output_dim = self.env.action_space.shape[0]
            self.policy = Policy_Continuous(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = output_dim)
        
        self.V = Value_Function_Apx()
       
        # self.policy = 
        # self.value_function


    def display_environment_basics(self):
        """
        """
        print("************************************")
        print("Basics of the environment\n")
        print("Action space:")
        action_space = self.env.action_space
        print(action_space)

        if isinstance(action_space, Box):
            print("Upper and lower bounds: ")
            print(action_space.high)
            print(action_space.low)

        print("Observation space:")
        print(self.env.observation_space)
        print("Upper and lower bounds: ")
        print(self.env.observation_space.high)
        print(self.env.observation_space.low)
        print("************************************")


    def run_episode(self, verbose = False, max_num_transitions=5000):
        """
        """
        episode_stats = defaultdict(list)
        observation = self.env.reset()
        for i in range(max_num_transitions):
            with torch.no_grad():
                if verbose:
                    print(f"Observation: {observation} ")

                action = self.policy.sample(torch.Tensor(observation)).detach().numpy()
                episode_stats['observations'].append(observation)
                episode_stats['actions'].append(action)
                observation, reward, done, info = self.env.step(action)
                episode_stats['rewards'].append(reward)
                if done:
                    break
        return episode_stats
    
    def compute_product_terms_for_policy_gradients_loss(self, episode_stats, product_terms_type = 'discounted_returns'):
        """
        episode_stats: dict containing episode history
        product_terms_type: whether to use 'transition_rewards', 'discounted_returns', 'td0_advantage', 'gae'
                    till now only discounted_returns has been implemented
        """
        episode_len = len(episode_stats['observations'])
        
        if product_terms_type == 'transition_rewards':
            episode_stats['product_terms']  = [sum(episode_stats[''])]* episode_len

        if product_terms_type == 'discounted_returns':
            # Calculate the Discounted Returns for each time step 
            episode_len = len(episode_stats['observations'])
            episode_stats['product_terms']  = [0]* episode_len
            for i in reversed(range(len(episode_stats['observations']))):
                if i == episode_len -1:
                    episode_stats['product_terms'][i] = episode_stats['rewards'][i]
                else:
                    episode_stats['product_terms'][i] = episode_stats['rewards'][i] + self.gamma * episode_stats['product_terms'][i+1]
            
        return episode_stats



    def run_vanilla_policy_gradients(self, num_iters, batch_size = 1000, product_terms_type = 'discounted_returns'):
        """
        num_iters: No of updates to make
        batch_size: No of transitions to collect before one update
        product_terms_type: whether to use 'transition_rewards', 'discounted_returns', 'td0_advantage', 'gae'
                            till now only 'discounted_returns' and 'transition_rewards' has been implemented
        """
        # Initialise the optimizers
        policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        # value_function_optim = optim.Adam(value_function.parameters)
        avg_returns = [] 
        batch_size = 5000 

        for iter in range(num_iters):
            # Right now one update comes from a single episode. Will change to a fixed number of transitions later.
            
            # Collection of transition info needed for policy gradient updates
            observations = [] 
            actions = [] 
            product_terms = [] 
            batch_full = False

            episode_lens = []
            episode_rewards  = [] 

            while(not batch_full):
                episode_stats = self.run_episode()
                episode_stats = self.compute_product_terms_for_policy_gradients_loss(episode_stats)

                episode_len = len(episode_stats['observations'])
                episode_reward_total = sum(episode_stats['rewards'])
                episode_lens.append(episode_len)
                episode_rewards.append(episode_reward_total)
                
                if episode_len + len(observations) > batch_size:
                    batch_full = True

                observations += episode_stats['observations']
                actions += episode_stats['actions']
                product_terms += episode_stats['product_terms']


            # Compute proxy policy loss (by running whose .backward(), we can get the correct gradients)
            actions = np.asarray(actions)
            observations = np.asarray(observations)
            product_terms = np.asarray(product_terms)
            # print(f'actions.shape : {actions.shape}')
            # print(f'observations.shape : {observations.shape}')
            # print(f'product_terms.shape : {product_terms.shape}')
            if self.policy_is_discrete:
                actions = torch.as_tensor(actions, dtype=torch.int32)
            else:
                actions = torch.as_tensor(actions, dtype=torch.float32)
            observations = torch.as_tensor(observations, dtype=torch.float32)
            product_terms = torch.as_tensor(product_terms, dtype=torch.float32)

            log_prob = self.policy.log_prob(observations, actions)
            
            policy_optimizer.zero_grad()
            policy_loss = torch.mean(-(log_prob * product_terms))
            policy_loss.backward()
            policy_optimizer.step()

            print(f"Iter: {iter}, loss: {policy_loss.item()}, average_epsiode_rewards: {np.average(episode_rewards)}, average_episode_lens: {np.average(episode_lens)}")
    
    def render_learnt_policy(self):
        for i in range(100):
            observation = self.env.reset()
            while(True):
                with torch.no_grad():
                    action = self.policy.sample(torch.Tensor(observation)).detach().numpy()
                    observation, reward, done, info = self.env.step(action)
                    self.env.render()
                    if done:
                        break



def run_vpg():
    # environment_name = 'CartPole-v0'
    # environment_name = 'Pendulum-v0'
    environment_name = 'MountainCarContinuous-v0'

    mdp = MDP(environment_name = environment_name, gamma = 1, hidden_dim = 32, horizon = 1000)
    mdp.run_vanilla_policy_gradients(50)
    mdp.render_learnt_policy()

    # episode_stats = mdp.run_episode()
    # print(episode_stats)





if __name__ == '__main__':
    run_vpg()