"""
Deep Q-Learning Training Script for Atari Environments
Formative 3 Assignment - ALU
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import os
import argparse
import numpy as np

# Register ALE environments
import ale_py
gym.register_envs(ale_py)

def create_atari_env(env_name, n_envs=4, seed=0):
    """
    Create and wrap Atari environment with proper preprocessing
    
    Args:
        env_name: Name of the Atari environment (e.g., 'BreakoutNoFrameskip-v4')
        n_envs: Number of parallel environments
        seed: Random seed
    """
    env = make_atari_env(env_name, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env

def train_dqn(
    env_name='BreakoutNoFrameskip-v4',
    policy='CnnPolicy',
    total_timesteps=1_000_000,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    buffer_size=100_000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    seed=0,
    save_path='./models/dqn_model',
    log_path='./logs/',
    experiment_name='dqn_atari'
):
    """
    Train a DQN agent on an Atari environment
    
    Args:
        env_name: Atari environment name
        policy: Policy type ('CnnPolicy' or 'MlpPolicy')
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        gamma: Discount factor
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        exploration_fraction: Fraction of training for exploration
        exploration_initial_eps: Initial epsilon for exploration
        exploration_final_eps: Final epsilon for exploration
        target_update_interval: Target network update frequency
        train_freq: Training frequency
        gradient_steps: Gradient steps per update
        seed: Random seed
        save_path: Path to save model
        log_path: Path for tensorboard logs
        experiment_name: Name for this experiment
    """
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create training environment
    print(f"\n{'='*60}")
    print(f"Creating {env_name} environment...")
    print(f"{'='*60}\n")
    env = create_atari_env(env_name, n_envs=4, seed=seed)
    
    # Create evaluation environment
    eval_env = create_atari_env(env_name, n_envs=1, seed=seed+1000)
    
    # Set up logger
    logger = configure(f"{log_path}/{experiment_name}", ["stdout", "csv", "tensorboard"])
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{os.path.dirname(save_path)}/best/",
        log_path=f"{log_path}/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{os.path.dirname(save_path)}/checkpoints/",
        name_prefix="dqn_checkpoint"
    )
    
    # Print hyperparameters
    print(f"\n{'='*60}")
    print("HYPERPARAMETERS:")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Policy: {policy}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Batch Size: {batch_size}")
    print(f"Buffer Size: {buffer_size:,}")
    print(f"Exploration Fraction: {exploration_fraction}")
    print(f"Initial Epsilon: {exploration_initial_eps}")
    print(f"Final Epsilon: {exploration_final_eps}")
    print(f"Target Update Interval: {target_update_interval}")
    print(f"Train Frequency: {train_freq}")
    print(f"Gradient Steps: {gradient_steps}")
    print(f"{'='*60}\n")
    
    # Create DQN model
    model = DQN(
        policy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=50000,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log=f"{log_path}/{experiment_name}",
        seed=seed
    )
    
    model.set_logger(logger)
    
    # Train the model
    print(f"\n{'='*60}")
    print("STARTING TRAINING...")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        progress_bar=True
    )
    
    # Save final model
    print(f"\n{'='*60}")
    print(f"Saving model to {save_path}")
    print(f"{'='*60}\n")
    model.save(save_path)
    
    # Save as dqn_model.zip (required by assignment)
    model.save("./dqn_model")
    print("Model also saved as ./dqn_model.zip")
    
    env.close()
    eval_env.close()
    
    return model

def run_experiment(experiment_num, **kwargs):
    """
    Run a single experiment with specified hyperparameters
    """
    experiment_name = f"experiment_{experiment_num:02d}"
    kwargs['experiment_name'] = experiment_name
    kwargs['save_path'] = f"./models/{experiment_name}/dqn_model"
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT {experiment_num}: {experiment_name}")
    print(f"{'#'*60}\n")
    
    model = train_dqn(**kwargs)
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT {experiment_num} COMPLETED")
    print(f"{'#'*60}\n")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN on Atari')
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',
                       help='Atari environment name')
    parser.add_argument('--policy', type=str, default='CnnPolicy',
                       choices=['CnnPolicy', 'MlpPolicy'],
                       help='Policy type')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=100_000,
                       help='Replay buffer size')
    parser.add_argument('--exp-fraction', type=float, default=0.1,
                       help='Exploration fraction')
    parser.add_argument('--eps-start', type=float, default=1.0,
                       help='Initial epsilon')
    parser.add_argument('--eps-end', type=float, default=0.01,
                       help='Final epsilon')
    parser.add_argument('--experiment', type=int, default=None,
                       help='Experiment number')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run training
    if args.experiment is not None:
        run_experiment(
            args.experiment,
            env_name=args.env,
            policy=args.policy,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            exploration_fraction=args.exp_fraction,
            exploration_initial_eps=args.eps_start,
            exploration_final_eps=args.eps_end,
            seed=args.seed
        )
    else:
        train_dqn(
            env_name=args.env,
            policy=args.policy,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            exploration_fraction=args.exp_fraction,
            exploration_initial_eps=args.eps_start,
            exploration_final_eps=args.eps_end,
            seed=args.seed
        )