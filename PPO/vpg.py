"""
Implement Vanilla Policy Gradients,  
using this Pseudocode: https://spinningup.openai.com/en/latest/algorithms/vpg.html

Versions: a. Returns as advantage b. TD0 error as advantage c. GAE as advantage
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

import gym 
from gym.spaces import Dict, Discrete, Box, Tuple

# np.random.seed(1)
# torch.manual_seed(1)
run_num = 1

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


class Policy_Continuous_v2():
    """
    Implements a Gaussian two layer neural network policy (Linear, tanh, Linear) for continuous action space. 

    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # super(Policy_Continuous_v2, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.mean_projection = nn.Linear(hidden_dim, output_dim)
        log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # Both mean and variance predicting neural nets share a layer. 
        self.action_mean_nn = nn.Sequential(self.layer1, nn.Tanh(), self.mean_projection)
    
    def parameters(self):
        return list(self.action_mean_nn.parameters()) + [self.log_std]
    
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
    def __init__(self, environment_name, gamma, hidden_dim, horizon, gae_lambda=0.97, activation = 'relu'):
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
            # self.policy = Policy_Continuous_v3(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = output_dim, activation = activation)
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
        episode_stats['last_time_step'] = [False]* len(episode_stats['observations'])
        episode_stats['last_time_step'][-1] = True
        return episode_stats
    
    def compute_product_terms_for_policy_gradients_loss(self, episode_stats, product_terms_type = 'discounted_returns'):
        """
        episode_stats: dict containing episode history
        product_terms_type: whether to use 'transition_rewards', 'discounted_returns', 'td0_advantage', 'gae'
                    till now only discounted_returns has been implemented
        """
        episode_len = len(episode_stats['observations'])

        episode_stats['discounted_returns']  = [0]* episode_len
        for i in reversed(range(len(episode_stats['observations']))):
            if i == episode_len -1:
                episode_stats['discounted_returns'][i] = episode_stats['rewards'][i]
            else:
                episode_stats['discounted_returns'][i] = episode_stats['rewards'][i] + self.gamma * episode_stats['discounted_returns'][i+1]

        
        if product_terms_type == 'transition_rewards':
            episode_stats['product_terms']  = [sum(episode_stats[''])]* episode_len

        if product_terms_type == 'discounted_returns':
            episode_stats['product_terms']  = episode_stats['discounted_returns']
        
        if product_terms_type == 'td0_advantage':
            with torch.no_grad():
                value_obs_curr_states = self.V(torch.as_tensor(np.asarray(episode_stats['observations']), dtype = torch.float32 )).detach().numpy()
            value_obs_next_states = np.zeros_like(value_obs_curr_states)
            value_obs_next_states[:-1] =  value_obs_curr_states[1:]
            episode_stats['product_terms'] = list( value_obs_curr_states + self.gamma *  value_obs_next_states - np.asarray(episode_stats['rewards']) )    

        if product_terms_type == 'gae':
            with torch.no_grad():
                value_obs_curr_states = self.V(torch.as_tensor(np.asarray(episode_stats['observations']), dtype = torch.float32 )).detach().numpy()
            value_obs_next_states = np.zeros_like(value_obs_curr_states)
            value_obs_next_states[:-1] =  value_obs_curr_states[1:]
            td_errors = value_obs_curr_states + self.gamma *  value_obs_next_states - np.asarray(episode_stats['rewards']) 
            gae = np.zeros_like(td_errors)
            for t in reversed(range(len(gae))):
                if t == len(gae)-1:
                    gae[t] = td_errors[t]
                else:
                    gae[t] = td_errors[t] + self.gamma * self.gae_lambda * gae[t+1]
            
            episode_stats['product_terms'] = list(gae)

        return episode_stats


    def update_policy(self, observations, actions, product_terms, policy_optimizer):
        log_prob = self.policy.log_prob(observations, actions)
        policy_optimizer.zero_grad()
        policy_loss = torch.mean(-(log_prob * product_terms))

        policy_loss.backward()
        policy_optimizer.step()
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
            last_time_step = []

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
                last_time_step += episode_stats['last_time_step']


            # Compute proxy policy loss (by running whose .backward(), we can get the correct gradients)
            actions = np.asarray(actions)
            observations = np.asarray(observations)
            product_terms = np.asarray(product_terms)
            discounted_returns = np.asarray(discounted_returns)


            if product_terms_type == 'gae' and normalize_gae:
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


def run_vpg():
    # environment_name = 'CartPole-v0'
    environment_name = 'Pendulum-v0'
    # environment_name = 'MountainCarContinuous-v0'
    wandb.init(project="Actor-Critics Methods for Continous Action spaces",  name=f"{environment_name}-discounted_returns_tanh_3_layers_32")
    config = wandb.config
    config.environment_name = environment_name
    config.gamma = 0.99
    config.hidden_dim = 32
    config.horizon = 1000
    config.num_value_optim_steps = 80
    config.product_terms_type = 'discounted_returns'
    config.run_num = run_num
    config.value_fn_lr = 1e-3
    config.num_layers = 2
    config.activation = 'tanh'
    config.normalize_gae = True

    mdp = MDP(environment_name = environment_name, gamma = 0.99, hidden_dim = 32, horizon = None, activation = 'tanh')
    mdp.run_vanilla_policy_gradients(num_iters = 10000, num_value_optim_steps = 80, product_terms_type = 'discounted_returns', model_name = 'discounted_returns_tanh_3_layers_32', value_fn_lr = 1e-3, normalize_gae = True)


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


def debug_environment():
    environment_name = 'MountainCarContinuous-v0'
    mdp = MDP(environment_name = environment_name, gamma = 0.99, hidden_dim = 32, horizon = 2000)
    w1 = torch.load(f'./checkpoints/{environment_name}/td0_advantage'+str(1999)+'-run'+str(run_num))['value_fn']
    w2 = torch.load(f'./checkpoints/{environment_name}/td0_advantage'+str(1990)+'-run'+str(run_num))['value_fn']

    print(w1['layer1.bias'])
    print(w2['layer1.bias'])

if __name__ == '__main__':
    run_vpg()
    # test_model_load()
    # test_model_parameters()
    # debug_environment()