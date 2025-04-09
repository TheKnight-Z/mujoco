import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# --- Define the toy double integrator environment ---
class DoubleIntegratorEnv:
    def __init__(self, dt=0.1, max_steps=200):
        self.dt = dt
        self.max_steps = max_steps
        self.action_limit = 1.0  # clip acceleration to [-1, 1]
        self.reset()
    
    def reset(self):
        self.state = np.array([0.0, 0.0])  # start at rest at the origin
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action):
        # Clip action
        action = np.clip(action, -self.action_limit, self.action_limit)
        pos, vel = self.state
        # Update dynamics: simple Euler integration
        new_vel = vel + action * self.dt
        new_pos = pos + vel * self.dt
        self.state = np.array([new_pos, new_vel])
        self.step_count += 1
        
        # Define a quadratic cost (reward is negative cost)
        reward = - (new_pos**2 + new_vel**2)
        done = self.step_count >= self.max_steps
        return self.state.copy(), reward, done, {}

# --- Define a simple PPO policy network ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        # Log std parameter for a Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.fc(state)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

# --- Define the PPO algorithm with reward logging ---
class PPO:
    def __init__(self, env, state_dim=2, action_dim=1, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=10):
        self.env = env
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        # For logging total reward per episode
        self.reward_log = []

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            log_probs = []
            rewards = []
            states = []
            actions = []
            
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, log_prob = self.policy.sample_action(state_tensor)
                next_state, reward, done, _ = self.env.step(action.item())
                
                states.append(state_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state
            
            # Log total reward for this episode
            total_reward = sum(rewards)
            self.reward_log.append(total_reward)
            
            returns = self.compute_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            states_tensor = torch.stack(states)
            actions_tensor = torch.stack(actions)
            old_log_probs = torch.stack(log_probs).squeeze()

            # PPO policy update
            for _ in range(self.epochs):
                mean, std = self.policy(states_tensor)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_tensor).squeeze()
                ratio = torch.exp(new_log_probs - old_log_probs)
                advantages = returns
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Loss = {loss.item():.4f}, Total Reward = {total_reward:.2f}")

        # After training, plot the reward curve
        plt.plot(self.reward_log)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward Curve during Training')
        plt.grid(True)
        plt.show()

# --- Main: Train the PPO agent on the double integrator ---
if __name__ == "__main__":
    # In an MJX setting, you would load your MuJoCo model and use mjx.put_model/data.
    env = DoubleIntegratorEnv()
    ppo_agent = PPO(env)
    ppo_agent.train(num_episodes=1000)