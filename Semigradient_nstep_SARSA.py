import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
cartpole = gym.make('CartPole-v1')
acrobat = gym.make('Acrobot-v1')

# action-value function parameterization
class QNetwork(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)


    def forward(self, state,action):
        x = torch.cat([state,action],0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def choose_action(Q,s,eps):
    Q_values = []
    for a in range(action_size):
        q_estimate_s_a = Q(torch.FloatTensor(s),torch.FloatTensor([a])).detach().numpy()
        Q_values.append(q_estimate_s_a)
        
    r = np.random.rand()
    if r<eps:
        a =  np.random.randint(action_size)
    else:
        a = np.argmax(Q_values)

    return a 

def SemiGradientNStepSARSA(env,gamma,N_episodes,eps,n):
    rewards_list = []
    for episode in range(N_episodes):
        
        # Generate an episode
    
        # starting state
        s, info = env.reset()
        a = choose_action(q_network,s,eps)
        T = np.inf
        states = [s]
        actions = [a]
        rewards = [0]
        t = 0
        while True: 
            if t<T:
                s1, r, terminated, truncated, info = env.step(actions[t])
                states.append(s1)
                rewards.append(r)
                if terminated or truncated:
                    T = t+1
                else:
                    a1 = choose_action(q_network,s1,eps)
                    actions.append(a1)

            tau = t-n+1

            if tau>=0:
                G = 0
                for i in range(tau+1,min(tau+n,T)+1):
                    G+= pow(gamma,i-tau-1)*rewards[i]
                if tau+n<T:
                    G = G + pow(gamma,n)*q_network(torch.FloatTensor(states[tau+n]),torch.FloatTensor([actions[tau+n]])).unsqueeze(0)
                else:
                    G = torch.tensor([G]).float().unsqueeze(0)
                
                 # Update value  
                q_estimate_s_a = q_network(torch.FloatTensor(states[tau]),torch.FloatTensor([actions[tau]])).unsqueeze(0)
                delta = G - q_estimate_s_a
                optimizer_value.zero_grad()
                value_loss = -delta.item()*q_estimate_s_a 
                value_loss.backward()
                optimizer_value.step()

            t+=1
            
            if tau>=T-1:
                break

    
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
eps = 0.4
n = 5

# Initialize environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_dims = 128

alpha_value = 1e-3


# run algo
trails = 5
trails_return = np.zeros((trails,N_episodes))
for i in range(trails):
    q_network = QNetwork(state_size+1,hidden_dims,1)
    optimizer_value = optim.Adam(q_network.parameters(), lr=alpha_value)
    
    trails_return[i] = SemiGradientNStepSARSA(env,gamma,N_episodes,eps,n)
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
plt.title('Semi gradient n step SARSA')
plt.savefig('Results/'+"{}_{}_{}_{}_{}_{}".format("acrobat","semigradientnstepsarsa","adam",alpha_value,eps,n)+'.png')
plt.show()

