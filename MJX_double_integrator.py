import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# Import MuJoCo and MJX modules
import mujoco
from mujoco import mjx
import jax.numpy as jnp

# --- Define the MJX double integrator environment ---
# We define a simple MuJoCo XML model for a double integrator.
DOUBLE_INTEGRATOR_XML = """
<mujoco model="double_integrator">
  <option timestep="0.1"/>
  <worldbody>
    <!-- A single body with a linear (slide) joint in the x-direction -->
    <body name="mass" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-10 10"/>
      <geom name="mass_geom" type="sphere" size="0.1" rgba="0.8 0.3 0.3 1"/>
    </body>
  </worldbody>
  <actuator>
    <!-- Apply a force along the slide joint -->
    <general name="force" joint="slider" gear="1"/>
  </actuator>
</mujoco>
"""

class MJXDoubleIntegratorEnv:
    def __init__(self, max_steps=200):
        self.max_steps = max_steps
        self.dt = 0.1
        # Load the MuJoCo model from the XML string
        self.model = mujoco.MjModel.from_xml_string(DOUBLE_INTEGRATOR_XML)
        self.data = mujoco.MjData(self.model)
        # Transfer the model and data to the accelerator via MJX.
        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, self.data)
        self.step_count = 0

    def reset(self):
        # Reset the underlying MuJoCo data.
        mujoco.mj_resetData(self.data)
        self.data.time = 0.0
        self.step_count = 0
        # Re-place data on device.
        self.mjx_data = mjx.put_data(self.model, self.data)
        # Get the current joint position (qpos) and velocity (qvel)
        # Here, qpos is a jax array; we convert it to numpy.
        state = np.array(self.mjx_data.qpos)
        return state

    def step(self, action):
        # In MuJoCo, the control input is applied via data.ctrl.
        # Here we set the control for the "force" actuator.
        # Note: action is expected to be a scalar.
        self.mjx_data = self.mjx_data.replace(ctrl=jnp.array([action]))
        # Advance the simulation one time step using MJX.
        self.mjx_data = mjx.step(self.mjx_model, self.mjx_data)
        self.step_count += 1
        # Get state: qpos (position) and qvel (velocity)
        state = np.array(self.mjx_data.qpos)
        # For reward, we use the same quadratic cost:
        pos = state[0]
        # qvel is a jax array; convert to numpy scalar
        vel = np.array(self.mjx_data.qvel)[0]
        reward = - (pos**2 + vel**2)
        done = self.step_count >= self.max_steps
        return state, reward, done, {}

# --- Define a simple PPO policy network (same as before) ---
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

# --- Define PPO with reward logging (same as before) ---
class PPO:
    def __init__(self, env, state_dim=1, action_dim=1, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=10):
        self.env = env
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
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
            state = self.env.reset()  # state is np.array of shape (1,)
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
            
            total_reward = sum(rewards)
            self.reward_log.append(total_reward)
            
            returns = self.compute_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            states_tensor = torch.stack(states)
            actions_tensor = torch.stack(actions)
            old_log_probs = torch.stack(log_probs).squeeze()

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
        
        plt.plot(self.reward_log)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward Curve during Training')
        plt.grid(True)
        plt.show()

# --- Main: Train the PPO agent with MJX simulation ---
if __name__ == "__main__":
    env = MJXDoubleIntegratorEnv(max_steps=200)
    # Note: state_dim is 1 because qpos is one-dimensional (the slide joint)
    ppo_agent = PPO(env, state_dim=1, action_dim=1)
    ppo_agent.train(num_episodes=1000)
