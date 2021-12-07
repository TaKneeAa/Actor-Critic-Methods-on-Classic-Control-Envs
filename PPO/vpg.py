"""
Implement Vanilla Policy Gradients,  
using this Pseudocode: https://spinningup.openai.com/en/latest/algorithms/vpg.html

Versions: a. Discounted Returns as advantage b. TD0 error as advantage c. GAE as advantage
Choices for Gaussian Policy: Policy_Continuous, Policy_Continuous_v2, Policy_Continuous_v3
"""
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

from policy_tools import Policy_Continuous, Policy_Continuous_v2, Policy_Continuous_v3

import gym 
from gym.spaces import Dict, Discrete, Box, Tuple
import argparse

# np.random.seed(1)
# torch.manual_seed(1)
run_num = 1


class Value_Function_Apx(nn.Module):
    """
    Implements a two layer neural network that approximates the Value Function
    """
    def __init__(self, input_dim, hidden_dim):
        super(Value_Function_Apx, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, observation):
        """
        """
        layer1_output = torch.tanh(self.layer1(observation))
        layer2_output = torch.squeeze(self.layer2(layer1_output))
        return layer2_output

class MDP():
    def __init__(self, environment_name, gamma, hidden_dim, horizon, gae_lambda=0.97, activation = 'relu', num_layers = 2):
        print(f"Initialising environment: {environment_name}")
        self.environment_name = environment_name
        self.env = gym.make(environment_name)
        if horizon:
            self.env._max_episode_steps = horizon
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.gae_lambda = gae_lambda
        self.input_dim = self.env.observation_space.shape[0]

        self.display_environment_basics()
        if isinstance(self.env.action_space, Discrete):
            self.policy_is_discrete = True
            output_dim = self.env.action_space.n
            self.policy = Policy_Discrete(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = output_dim)
        else:
            self.policy_is_discrete = False
            output_dim = self.env.action_space.shape[0]

            if num_layers == 3:
                self.policy = Policy_Continuous_v3(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = output_dim, activation = activation)
            elif num_layers == 2:
                self.policy = Policy_Continuous_v2(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = output_dim)

        self.V = Value_Function_Apx(input_dim = self.input_dim, hidden_dim = self.hidden_dim)


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
        
        # episode_stats['last_time_step'] = [False]* len(episode_stats['observations'])
        # episode_stats['last_time_step'][-1] = True
        return episode_stats
    
    def compute_product_terms_for_policy_gradients_loss(self, episode_stats, product_terms_type = 'discounted_returns'):
        """
        episode_stats: dict containing episode history
        product_terms_type: whether to use 'transition_rewards', 'discounted_returns', 'td0_advantage', 'gae'
                    till now only discounted_returns has been implemented
        """
        episode_len = len(episode_stats['observations'])

        if product_terms_type == 'discounted_returns' or product_terms_type == 'td0_advantage':
            episode_stats['discounted_returns']  = [0]* episode_len
            for i in reversed(range(len(episode_stats['observations']))):
                if i == episode_len -1:
                    episode_stats['discounted_returns'][i] = episode_stats['rewards'][i]
                else:
                    episode_stats['discounted_returns'][i] = episode_stats['rewards'][i] + self.gamma * episode_stats['discounted_returns'][i+1]

        if product_terms_type == 'transition_rewards':
            episode_stats['product_terms']  = [sum(episode_stats['rewards'])]* episode_len

        if product_terms_type == 'discounted_returns':
            episode_stats['product_terms']  = episode_stats['discounted_returns']
        
        if product_terms_type == 'td0_advantage':
            with torch.no_grad():
                value_obs_curr_states = self.V(torch.as_tensor(np.asarray(episode_stats['observations'], dtype = np.float32), dtype = torch.float32 )).detach().numpy()
            value_obs_next_states = np.zeros_like(value_obs_curr_states)
            value_obs_next_states[:-1] =  value_obs_curr_states[1:]
            episode_stats['product_terms'] = list( np.asarray(episode_stats['rewards'], dtype = np.float32)  + self.gamma *  value_obs_next_states - value_obs_curr_states)    

        if product_terms_type == 'gae':
            with torch.no_grad():
                value_obs_curr_states = self.V(torch.as_tensor(np.asarray(episode_stats['observations'], dtype = np.float32), dtype = torch.float32 )).detach().numpy()
            value_obs_next_states = np.zeros_like(value_obs_curr_states)
            value_obs_next_states[:-1] =  value_obs_curr_states[1:]
            td_errors = np.asarray(episode_stats['rewards']) + self.gamma *  value_obs_next_states - value_obs_curr_states
            gae = np.zeros_like(td_errors)
            for t in reversed(range(len(gae))):
                if t == len(gae)-1:
                    gae[t] = td_errors[t]
                else:
                    gae[t] = td_errors[t] + self.gamma * self.gae_lambda * gae[t+1]
            episode_stats['product_terms'] = list(gae)

        return episode_stats

    def update_policy(self, observations, actions, product_terms, policy_optimizer):
        policy_optimizer.zero_grad()
        log_prob = self.policy.log_prob(observations, actions)
        policy_loss = torch.mean(-(log_prob * product_terms))
        # print(policy_loss)
        # print(list(self.policy.parameters()))
        policy_loss.backward()
        # for param in self.policy.parameters():
        #     print('grad')
        #     print(param.grad)
        policy_optimizer.step()
        # print(list(self.policy.parameters()))
        policy_loss_numpy = policy_loss.item()
        return policy_loss_numpy
    
    def update_value_fn_n_steps(self, observations, returns, value_fn_optimizer, num_value_optim_steps):
        """
        """
        for i in range(num_value_optim_steps):
            value_fn_optimizer.zero_grad()
            loss = torch.mean(torch.square(self.V(observations) - returns))
            loss.backward()
            value_fn_optimizer.step()
        
        with torch.no_grad():
            return torch.mean(torch.square(self.V(observations) - returns)).detach().numpy()



    def run_vanilla_policy_gradients(self, num_iters, num_value_optim_steps, batch_size = 4000, product_terms_type = 'discounted_returns', model_name = 'default', policy_lr = 1e-4, value_fn_lr = 1e-3, normalize_gae = True):
        """
        num_iters: No of updates to make
        batch_size: No of transitions to collect before one update
        product_terms_type: whether to use 'transition_rewards', 'discounted_returns', 'td0_advantage', 'gae'
                            till now only 'discounted_returns' and 'transition_rewards' has been implemented
        num_value_optim_steps: 
        normalize_gae: either normalise to zero mean, unit std or else just center it.
        """
        # Initialise the optimizers
        policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        value_fn_optimizer = optim.Adam(self.V.parameters(), lr=value_fn_lr)
        avg_returns = [] 
        batch_size = 5000 
        num_episodes = 0 
        for iter in range(num_iters):
            # Right now one update comes from a single episode. Will change to a fixed number of transitions later.
            
            # Collection of transition info needed for policy gradient updates
            observations = [] 
            actions = [] 
            product_terms = [] 
            discounted_returns = [] 
            # last_time_step = []

            batch_full = False

            episode_lens = []
            episode_rewards  = [] 

            while(not batch_full):
                episode_stats = self.run_episode()
                num_episodes+=1
                episode_stats = self.compute_product_terms_for_policy_gradients_loss(episode_stats, product_terms_type = product_terms_type)

                episode_len = len(episode_stats['observations'])
                episode_reward_total = sum(episode_stats['rewards'])
                episode_lens.append(episode_len)
                episode_rewards.append(episode_reward_total)
                wandb.log({f"{self.environment_name} {model_name} Reward Per Episode" :episode_reward_total},step=num_episodes)
                # print(f"Reward Per Episode :{episode_reward_total}")
                if episode_len + len(observations) > batch_size:
                    batch_full = True


                observations += episode_stats['observations']
                actions += episode_stats['actions']
                product_terms += episode_stats['product_terms']
                discounted_returns += episode_stats['discounted_returns']
                # last_time_step += episode_stats['last_time_step']


            # Compute proxy policy loss (by running whose .backward(), we can get the correct gradients)
            actions = np.asarray(actions)
            observations = np.asarray(observations)
            product_terms = np.asarray(product_terms)
            discounted_returns = np.asarray(discounted_returns)


            if product_terms_type == 'gae':
                if normalize_gae:
                    product_terms = (product_terms - np.mean(product_terms)) / np.std(product_terms)
                else:
                    product_terms = product_terms - np.mean(product_terms)

            if self.policy_is_discrete:
                actions = torch.as_tensor(actions, dtype=torch.int32)
            else:
                actions = torch.as_tensor(actions, dtype=torch.float32)

            observations = torch.as_tensor(observations, dtype=torch.float32)
            product_terms = torch.as_tensor(product_terms, dtype=torch.float32)
            discounted_returns = torch.as_tensor(discounted_returns, dtype=torch.float32)

            policy_loss_numpy = self.update_policy(observations, actions, product_terms, policy_optimizer)
            value_fn_loss_after_updates = self.update_value_fn_n_steps(observations, discounted_returns, value_fn_optimizer, num_value_optim_steps)

            print(f"Iter: {iter}, policy loss: {policy_loss_numpy}, value_fn loss : {value_fn_loss_after_updates}")
            wandb.log({f"{self.environment_name} {model_name} Actor loss ":policy_loss_numpy},step=num_episodes)
            wandb.log({f"{self.environment_name} {model_name} Critic loss ":value_fn_loss_after_updates},step=num_episodes)

            if iter%500 == 0 or iter == num_iters-1:
                torch.save({
                'iter': iter,
                'policy_log_std': self.policy.log_std,
                'policy_action_mean_nn': self.policy.action_mean_nn.state_dict(),
                'value_fn': self.V.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'value_fn_optimizer': value_fn_optimizer.state_dict(),
                }, f'./checkpoints/{self.environment_name}/{model_name}'+str(iter)+'-run'+str(run_num))  


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
    
    def load_model_from_file(self, model_file):
        weights_dict = torch.load(model_file)
        self.V.load_state_dict(weights_dict['value_fn'])
        self.policy.action_mean_nn.load_state_dict(weights_dict['policy_action_mean_nn'])
        self.policy.log_std = weights_dict['policy_log_std']
        print("Loaded!")

def run_vpg(args):
    model_name = f'{args.product_terms_type}_{args.activation}_{args.num_layers}_{args.hidden_dim}'

    args.normalize_gae = args.normalize_gae.lower() == 'true'
    wandb.init(project="Actor-Critics Methods for Continous Action spaces",  name=f"{environment_name}-{model_name}")
    # environment_name = 'MountainCarContinuous-v0'
    config = wandb.config
    config.environment_name = args.environment_name
    config.gamma = args.gamma
    config.hidden_dim = args.hidden_dim
    config.horizon = args.horizon
    config.num_value_optim_steps = args.num_value_optim_steps
    config.product_terms_type = args.product_terms_type
    config.run_num = args.run_num
    config.value_fn_lr = args.value_fn_lr
    config.policy_lr = args.policy_lr
    config.num_layers = args.num_layers
    config.activation = args.activation
    config.normalize_gae = args.normalize_gae

    mdp = MDP(environment_name = args.environment_name, gamma = args.gamma, hidden_dim = args.hidden_dim, horizon = args.horizon, activation = args.activation, num_layers = args.num_layers)
    mdp.run_vanilla_policy_gradients(num_iters = args.num_iters, num_value_optim_steps = args.num_value_optim_steps, product_terms_type = args.product_terms_type, model_name = model_name, value_fn_lr = args.value_fn_lr, policy_lr = args.policy_lr, normalize_gae = args.normalize_gae)

def test_model_load():
    environment_name = 'MountainCarContinuous-v0'
    mdp = MDP(environment_name = environment_name, gamma = 0.99, hidden_dim = 32, horizon = 2000)
    mdp.load_model_from_file(f'./checkpoints/{environment_name}/td0_advantage'+str(1999)+'-run'+str(run_num))
    mdp.render_learnt_policy()


def test_model_parameters():
    environment_name = 'MountainCarContinuous-v0'
    mdp = MDP(environment_name = environment_name, gamma = 0.99, hidden_dim = 32, horizon = None, activation = 'relu')
    # mdp.load_model_from_file(f'./checkpoints/{environment_name}/td0_advantage'+str(1999)+'-run'+str(run_num))
    print([elem.shape for elem in mdp.policy.parameters()])

def debug_gradient_update():
    environment_name = 'MountainCarContinuous-v0'
    mdp = MDP(environment_name = environment_name, gamma = 0.99, hidden_dim = 32, horizon = 2000)
    w1 = torch.load(f'./checkpoints/{environment_name}/td0_advantage'+str(1999)+'-run'+str(run_num))['value_fn']
    w2 = torch.load(f'./checkpoints/{environment_name}/td0_advantage'+str(1990)+'-run'+str(run_num))['value_fn']

    print(w1['layer1.bias'])
    print(w2['layer1.bias'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default='5000')
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--num_value_optim_steps', type=int, default=80)
    parser.add_argument('--product_terms_type', type=str, default='discounted_returns')
    parser.add_argument('--run_num', type=int, default=1)
    parser.add_argument('--value_fn_lr', type=int, default=1e-3)
    parser.add_argument('--policy_lr', type=int, default=1e-2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--normalize_gae', type=str, default='True')
    args = parser.parse_args()
    run_vpg(args)
