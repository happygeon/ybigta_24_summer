import argparse
import gymnasium as gym
import torch
from itertools import count
from tqdm import tqdm
from assets import A2C, device

def main(args):
    env = gym.make('LunarLander-v2', render_mode='None')
    state_size = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    agent = A2C(state_size, action_size, args.gamma, args.lr, args.lr, args.tau)

    for i_episode in tqdm(range(args.episodes)):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for t in count():
            action, value = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if done and terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(state, action, value, next_state, reward)

            state = next_state

            if done:
                if terminated and total_reward >= 200:  # 성공적인 착지로 간주
                    print(f"Episode {i_episode}: Successful landing! Total reward: {total_reward}")
                else:
                    print(f"Episode {i_episode}: Crash or failure. Total reward: {total_reward}")
                agent.episode_rewards.append(total_reward)
                agent.plot_rewards()
                break
            
            if agent.memory.length >= args.update_step:
                agent.update()

    print('Complete')

    # Save the model
    agent.actor.to('cpu')
    agent.critic.to('cpu')
    print('Saving model...')
    torch.save(agent.actor.state_dict(), args.save_path + '/actor.pth')
    torch.save(agent.critic.state_dict(), args.save_path + '/critic.pth')
    agent.actor.to(device)
    agent.critic.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train the agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target network')
    parser.add_argument('--update_step', type=int, default=10, help='Number of steps between updates')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save the trained model')
    args = parser.parse_args()
    main(args)
