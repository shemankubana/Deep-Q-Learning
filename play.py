"""
Deep Q-Learning Evaluation Script for Atari Assault
Formative 3 Assignment - ALU
"""

import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import argparse
import time

# Register ALE environments
import ale_py
gym.register_envs(ale_py)

def create_atari_env_single(env_name, seed=0, render_mode='human'):
    """
    Create single Atari environment for visualization
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
    
    # Frame stacking for temporal information
    try:
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    except AttributeError:
        env = gym.wrappers.FrameStack(env, num_stack=4)
    
    return env

def play_game(
    model_path='./dqn_model.zip',
    env_name='ALE/Assault-v5',
    n_episodes=5,
    render=True,
    deterministic=True,
    seed=0,
    delay=0.01
):
    """
    Load trained DQN model and evaluate on Assault
    Uses GreedyQPolicy (deterministic=True) for evaluation
    """
    
    print(f"\n{'='*60}")
    print("LOADING TRAINED MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Policy: {'Greedy (deterministic)' if deterministic else 'Stochastic'}")
    print(f"{'='*60}\n")
    
    # Load trained model
    try:
        model = DQN.load(model_path)
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you have trained the model first using train.py")
        return
    
    # Create environment
    render_mode = 'human' if render else 'rgb_array'
    env = create_atari_env_single(env_name, seed=seed, render_mode=render_mode)
    
    # Statistics
    all_episode_rewards = []
    all_episode_lengths = []
    
    print(f"{'='*60}")
    print("STARTING GAMEPLAY")
    print(f"{'='*60}\n")
    
    # Play episodes
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed+episode)
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        print(f"Episode {episode + 1}/{n_episodes}")
        print("-" * 40)
        
        while not (done or truncated):
            # GreedyQPolicy: select action with highest Q-value
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(delay)
        
        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward}")
        print(f"  Length: {episode_length} steps")
        print()
    
    env.close()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {n_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(all_episode_rewards):.2f}")
    print(f"  Std:  {np.std(all_episode_rewards):.2f}")
    print(f"  Min:  {np.min(all_episode_rewards):.2f}")
    print(f"  Max:  {np.max(all_episode_rewards):.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean: {np.mean(all_episode_lengths):.2f}")
    print(f"  Std:  {np.std(all_episode_lengths):.2f}")
    print(f"  Min:  {np.min(all_episode_lengths):.0f}")
    print(f"  Max:  {np.max(all_episode_lengths):.0f}")
    print(f"{'='*60}\n")
    
    return all_episode_rewards, all_episode_lengths

def record_video(
    model_path='./dqn_model.zip',
    env_name='ALE/Assault-v5',
    n_episodes=3,
    video_folder='./videos',
    video_length=0,
    seed=0
):
    """
    Record video of trained agent playing Assault
    """
    from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
    import os

    print(f"\n{'='*60}")
    print("RECORDING VIDEO")
    print(f"{'='*60}\n")

    os.makedirs(video_folder, exist_ok=True)
    print(f"Videos will be saved to: {os.path.abspath(video_folder)}\n")

    # Load model
    model = DQN.load(model_path)
    
    # Create environment
    def make_env():
        env = gym.make(env_name, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
        
        try:
            env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        except AttributeError:
            env = gym.wrappers.FrameStack(env, num_stack=4)
        
        return env
    
    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length if video_length > 0 else 100000,
        name_prefix=f"dqn_assault"
    )

    obs = env.reset()
    episode_count = 0

    print(f"Recording {n_episodes} episode(s)...")

    while episode_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        if dones[0]:
            episode_count += 1
            print(f"Episode {episode_count}/{n_episodes} recorded")
            if episode_count < n_episodes:
                obs = env.reset()

    env.close()
    print(f"\n✓ Videos saved to {os.path.abspath(video_folder)}")
    print(f"✓ Recorded {episode_count} episode(s)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play trained DQN agent on Assault')
    parser.add_argument('--model-path', type=str, default='./dqn_model.zip',
                       help='Path to trained model')
    parser.add_argument('--env', type=str, default='ALE/Assault-v5',
                       help='Atari environment name')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to play')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy instead of greedy')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--delay', type=float, default=0.01,
                       help='Delay between frames (seconds)')
    parser.add_argument('--record', action='store_true',
                       help='Record video instead of playing')
    parser.add_argument('--video-folder', type=str, default='./videos',
                       help='Folder to save recorded videos')
    
    args = parser.parse_args()
    
    if args.record:
        record_video(
            model_path=args.model_path,
            env_name=args.env,
            n_episodes=args.episodes,
            video_folder=args.video_folder,
            seed=args.seed
        )
    else:
        play_game(
            model_path=args.model_path,
            env_name=args.env,
            n_episodes=args.episodes,
            render=not args.no_render,
            deterministic=not args.stochastic,
            seed=args.seed,
            delay=args.delay
        )