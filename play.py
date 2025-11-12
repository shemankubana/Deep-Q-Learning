"""
Deep Q-Learning Evaluation/Playing Script for Atari Environments
Formative 3 Assignment - ALU
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import argparse
import time

# Register ALE environments
import ale_py
gym.register_envs(ale_py)

def create_atari_env_single(env_name, seed=0, render_mode='human'):
    """
    Create single Atari environment for visualization
    
    Args:
        env_name: Name of the Atari environment
        seed: Random seed
        render_mode: Rendering mode ('human' for GUI, 'rgb_array' for recording)
    """
    # Create environment with rendering enabled
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
    
    # Use FrameStackObservation (new name in gymnasium)
    try:
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    except AttributeError:
        # Fallback for older gymnasium versions
        env = gym.wrappers.FrameStack(env, num_stack=4)
    
    return env

def play_game(
    model_path='./dqn_model.zip',
    env_name='BreakoutNoFrameskip-v4',
    n_episodes=5,
    render=True,
    deterministic=True,
    seed=0,
    delay=0.01
):
    """
    Load trained DQN model and play Atari game
    
    Args:
        model_path: Path to saved model
        env_name: Atari environment name
        n_episodes: Number of episodes to play
        render: Whether to render the game
        deterministic: Whether to use deterministic (greedy) policy
        seed: Random seed
        delay: Delay between frames (seconds)
    """
    
    print(f"\n{'='*60}")
    print("LOADING TRAINED MODEL")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Number of episodes: {n_episodes}")
    print(f"Deterministic policy: {deterministic}")
    print(f"{'='*60}\n")
    
    # Load the trained model
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
            # Use greedy policy (deterministic=True) for best performance
            action, _states = model.predict(obs, deterministic=deterministic)  # type: ignore
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(delay)  # Add small delay for visualization
        
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
    print(f"Episodes played: {n_episodes}")
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
    env_name='BreakoutNoFrameskip-v4',
    n_episodes=3,
    video_folder='./videos',
    video_length=0,  # 0 means record entire episode
    seed=0
):
    """
    Record video of trained agent playing
    
    Args:
        model_path: Path to saved model
        env_name: Atari environment name
        n_episodes: Number of episodes to record
        video_folder: Folder to save videos
        video_length: Length of video in steps (0 for full episode)
        seed: Random seed
    """
    from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
    
    print(f"\n{'='*60}")
    print("RECORDING VIDEO")
    print(f"{'='*60}\n")
    
    # Load model
    model = DQN.load(model_path)
    
    # Create environment
    def make_env():
        env = gym.make(env_name, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env)
        
        # Use FrameStackObservation (new name in gymnasium)
        try:
            env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        except AttributeError:
            # Fallback for older gymnasium versions
            env = gym.wrappers.FrameStack(env, num_stack=4)
        
        return env
    
    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"dqn_{env_name}"
    )
    
    obs = env.reset()
    
    for _ in range(n_episodes * 1000):  # Play long enough to get n_episodes
        action, _ = model.predict(obs, deterministic=True)  # type: ignore
        obs, _, dones, _ = env.step(action)
        
        if dones[0]:
            obs = env.reset()
    
    env.close()
    print(f"✓ Video saved to {video_folder}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play trained DQN agent')
    parser.add_argument('--model-path', type=str, default='./dqn_model.zip',
                       help='Path to trained model')
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5',
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