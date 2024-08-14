import math
import random
from collections import namedtuple, deque
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####### 여기서부터 코드를 작성하세요 #######
# Actor 신경망을 구현해주세요!
class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

# Critic 신경망을 구현해주세요!
class Critic(nn.Module):
    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

####### 여기까지 코드를 작성하세요 #######
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.length = 0

    def push(self, state, action, value, next_state, reward):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.length += 1

    def clear(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.length = 0


class A2C:
    def __init__(self, state_size, action_size, gamma, lr_actor, lr_critic, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.episode_rewards = []
        self.memory = Memory()

        # 플로팅 초기화
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.reward_line, = self.ax.plot([], [], label='Total Reward')
        self.ax.legend()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')

    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
            action = torch.multinomial(action_probs, num_samples=1)
            value = self.critic(state)
        return action, value

    def plot_rewards(self):
        self.reward_line.set_xdata(range(len(self.episode_rewards)))
        self.reward_line.set_ydata(self.episode_rewards)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    ####### 여기서부터 코드를 작성하세요 #######
    # Actor-Critic의 업데이트를 구현해주세요!
    def update(self):
        states = torch.stack(self.memory.states, dim = 0).to(device)
        actions = torch.tensor(self.memory.actions, dtype=torch.int64).to(device)
        values = torch.tensor(self.memory.values, dtype=torch.float32).to(device)
        
        default_tensor = torch.zeros((1, 8), dtype=torch.float32)
        next_states_list = [s if s is not None else default_tensor for s in self.memory.next_states]
        next_states = torch.stack(next_states_list).to(device)
        
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32).to(device)

        # Critic 업데이트
        self.optimizer_critic.zero_grad()
        target_values = rewards + self.gamma * self.critic(next_states).squeeze()
        critic_loss = F.mse_loss(self.critic(states).squeeze(), target_values)
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor 업데이트
        self.optimizer_actor.zero_grad()
        advantages = (target_values - values).detach()
        action_probs = self.actor(states)
        log_action_probs = torch.log(action_probs[range(len(actions)), 0, actions])
        actor_loss = -torch.mean(log_action_probs * advantages)
        actor_loss.backward()
        self.optimizer_actor.step()

        self.memory.clear()
    ####### 여기까지 코드를 작성하세요 #######