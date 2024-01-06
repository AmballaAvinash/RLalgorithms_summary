import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
cartpole = gym.make('CartPole-v1')
acrobat = gym.make('Acrobot-v1')


# policy parameterization
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), -1)
        return x

# state-value function parameterization
class ValueNetwork(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x


def Reinforce(env,gamma,N_episodes):
    rewards_list = []
    for episode in range(N_episodes):

        # Generate an episode
        states = []
        actions = []
        rewards = []
            
        # Generate an episode
    
        # starting state
        s, info = env.reset()
        T = 0    
        while True: 
            prob_action = policy_network(torch.FloatTensor(s))
            a = np.random.choice(action_size, p=prob_action.detach().numpy())
            
            s1, r, terminated, truncated, info = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            
            s = s1
            T+=1
            if terminated or truncated:
                    break


        for t in range(T):

            # calculate G
            G = 0
            for i in range(t,T):
                G+= pow(gamma,i-t)*rewards[i]

            G = torch.tensor([G]).float().unsqueeze(0)
            
            # value function
            value_estimate = value_network(torch.FloatTensor(states[t])).unsqueeze(0)
            delta = G - value_estimate
    
            # Update value
            optimizer_value.zero_grad()
            value_loss = -value_estimate * delta.item()   
            value_loss.backward()
            optimizer_value.step()
            
            # Update policy
            optimizer_policy.zero_grad()
            prob_action = policy_network(torch.FloatTensor(states[t]))
            log_pi_s_a = torch.log(prob_action[actions[t]])
            policy_loss = -pow(gamma,t)*log_pi_s_a * delta.item()  
            policy_loss.backward()
            optimizer_policy.step()
        
        
        env.close()
        
        # Store rewards
        sum_reward = np.sum(rewards)
        rewards_list.append(sum_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Sum Reward: {sum_reward}")

    return rewards_list
    


# Training loop

env = acrobat
N_episodes = 2000
gamma = 0.99  # Discount factor

# Initialize environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_dims = 128

alpha_policy = 1e-4
alpha_value = 1e-3

# run algo
trails = 5
trails_return = np.zeros((trails,N_episodes))
for i in range(trails):
    policy_network = PolicyNetwork(state_size, hidden_dims, action_size)
    value_network = ValueNetwork(state_size,hidden_dims,1)
    
    optimizer_policy = optim.Adam(policy_network.parameters(), lr=alpha_policy)
    optimizer_value = optim.Adam(value_network.parameters(), lr=alpha_value)

    trails_return[i] = Reinforce(env,gamma,N_episodes)
    # plotting reward
    # mean_reward = []
    # for i in range(len(rewards_list)):
    #     mean_reward.append(np.mean(rewards_list[0:i]))
    # plt.plot(rewards_list)
    # plt.plot(mean_reward)
    # plt.xlabel('Episode')
    # plt.ylabel('Total reward on episode')
    # plt.title('Reinforce')
    # plt.show()

x = np.arange(N_episodes)
y = np.mean(trails_return,axis=0)
std = np.std(trails_return,axis=0)

# Plot the mean values as a line
plt.plot(x, y, label='Mean')

# Add error bars for the standard deviations
# plt.errorbar(x, y, yerr=std, color='red',  linestyle='-', capsize=2, label='Std Dev')

plt.xlabel("Episode")
plt.ylabel("Total Average reward over trails")
plt.title('Reinforce')
plt.savefig('Results/'+"{}_{}_{}_{}_{}".format("acrobat","reinforce","adam",alpha_policy,alpha_value)+'.png')
plt.show()


