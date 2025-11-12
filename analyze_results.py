"""
Results Analysis and Visualization Script
Analyzes training logs and generates comparison plots
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_experiment_results(log_dir='./logs'):
    """
    Load results from all experiments
    """
    results = {}
    
    for exp_num in range(1, 11):
        exp_name = f"experiment_{exp_num:02d}"
        exp_path = Path(log_dir) / exp_name
        
        if exp_path.exists():
            # Try to load monitor CSV files
            monitor_files = list(exp_path.glob("**/monitor.csv"))
            if monitor_files:
                try:
                    # Read monitor CSV
                    df = pd.read_csv(monitor_files[0], skiprows=1)
                    results[exp_num] = {
                        'name': exp_name,
                        'rewards': df['r'].values if 'r' in df.columns else [],
                        'lengths': df['l'].values if 'l' in df.columns else [],
                        'times': df['t'].values if 't' in df.columns else []
                    }
                except Exception as e:
                    print(f"Warning: Could not load {exp_name}: {e}")
    
    return results

def plot_reward_comparison(results, save_path='images/reward_comparison.png'):
    """
    Plot reward curves for all experiments
    """
    plt.figure(figsize=(14, 8))
    
    for exp_num, data in results.items():
        rewards = data['rewards']
        if len(rewards) > 0:
            # Calculate rolling mean for smoother curves
            window = min(100, len(rewards) // 10)
            if window > 1:
                rewards_smooth = pd.Series(rewards).rolling(window=window).mean()
                plt.plot(rewards_smooth, label=f"Exp {exp_num}: {data['name']}", alpha=0.7)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Reward Comparison Across All Experiments', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved reward comparison to {save_path}")
    plt.close()

def plot_episode_lengths(results, save_path='images/episode_lengths.png'):
    """
    Plot episode length trends
    """
    plt.figure(figsize=(14, 8))
    
    for exp_num, data in results.items():
        lengths = data['lengths']
        if len(lengths) > 0:
            # Calculate rolling mean
            window = min(100, len(lengths) // 10)
            if window > 1:
                lengths_smooth = pd.Series(lengths).rolling(window=window).mean()
                plt.plot(lengths_smooth, label=f"Exp {exp_num}", alpha=0.7)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Length (steps)', fontsize=12)
    plt.title('Episode Length Trends', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved episode lengths to {save_path}")
    plt.close()

def plot_final_performance(results, save_path='images/final_performance.png'):
    """
    Bar plot of final average performance
    """
    exp_nums = []
    avg_rewards = []
    std_rewards = []
    
    for exp_num, data in results.items():
        rewards = data['rewards']
        if len(rewards) > 0:
            # Take last 100 episodes
            last_rewards = rewards[-100:]
            exp_nums.append(exp_num)
            avg_rewards.append(np.mean(last_rewards))
            std_rewards.append(np.std(last_rewards))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(exp_nums)), avg_rewards, yerr=std_rewards, 
                    capsize=5, alpha=0.7, color='steelblue')
    
    # Color the best performing experiment
    if avg_rewards:
        best_idx = np.argmax(avg_rewards)
        bars[best_idx].set_color('gold')
    
    plt.xlabel('Experiment Number', fontsize=12)
    plt.ylabel('Average Reward (Last 100 Episodes)', fontsize=12)
    plt.title('Final Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(exp_nums)), [f"Exp {n}" for n in exp_nums])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved final performance to {save_path}")
    plt.close()

def generate_summary_table(results, save_path='experiment_summary.csv'):
    """
    Generate summary statistics table
    """
    summary_data = []
    
    for exp_num, data in results.items():
        rewards = data['rewards']
        lengths = data['lengths']
        
        if len(rewards) > 0:
            # Calculate statistics
            last_100_rewards = rewards[-100:]
            summary_data.append({
                'Experiment': exp_num,
                'Name': data['name'],
                'Total Episodes': len(rewards),
                'Final Avg Reward': np.mean(last_100_rewards),
                'Final Std Reward': np.std(last_100_rewards),
                'Max Reward': np.max(rewards),
                'Avg Episode Length': np.mean(lengths) if len(lengths) > 0 else 0,
                'Max Episode Length': np.max(lengths) if len(lengths) > 0 else 0
            })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Final Avg Reward', ascending=False)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\n✓ Saved summary table to {save_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df

def generate_markdown_table(summary_df, config_file='experiment_configs.json'):
    """
    Generate markdown table for README
    """
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
    except:
        print("Warning: Could not load experiment configurations")
        configs = []
    
    markdown = "\n## 📊 Experiment Results Summary\n\n"
    markdown += "| Exp | Name | Avg Reward | Std | Max | Hyperparameters | Observed Behavior |\n"
    markdown += "|-----|------|------------|-----|-----|-----------------|-------------------|\n"
    
    for _, row in summary_df.iterrows():
        exp_num = int(row['Experiment'])
        
        # Find config
        config = next((c for c in configs if c['id'] == exp_num), None)
        if config:
            params = config['params']
            params_str = f"lr={params['lr']}, γ={params['gamma']}, batch={params['batch_size']}"
            behavior = config.get('expected_behavior', 'N/A')
        else:
            params_str = "N/A"
            behavior = "N/A"
        
        markdown += f"| {exp_num} | {row['Name']} | {row['Final Avg Reward']:.2f} | "
        markdown += f"{row['Final Std Reward']:.2f} | {row['Max Reward']:.2f} | "
        markdown += f"{params_str} | {behavior} |\n"
    
    # Save to file
    with open('experiment_results_table.md', 'w') as f:
        f.write(markdown)
    
    print("✓ Generated markdown table in experiment_results_table.md")
    print("\nCopy this table to your README.md file!\n")

def main():
    """
    Main analysis function
    """
    print("\n" + "="*80)
    print("DQN EXPERIMENT RESULTS ANALYSIS")
    print("="*80 + "\n")
    
    # Load results
    print("Loading experiment results...")
    results = load_experiment_results('./logs')
    
    if not results:
        print("✗ No experiment results found in ./logs directory")
        print("Make sure you have run experiments and they generated monitor.csv files")
        return
    
    print(f"✓ Loaded {len(results)} experiments\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_reward_comparison(results)
    plot_episode_lengths(results)
    plot_final_performance(results)
    
    # Generate summary
    print("\nGenerating summary statistics...")
    summary_df = generate_summary_table(results)
    
    # Generate markdown table
    print("\nGenerating markdown table...")
    generate_markdown_table(summary_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • images/reward_comparison.png")
    print("  • images/episode_lengths.png")
    print("  • images/final_performance.png")
    print("  • experiment_summary.csv")
    print("  • experiment_results_table.md")
    print("\nNext steps:")
    print("  1. Review the plots in the images/ directory")
    print("  2. Copy the markdown table to your README.md")
    print("  3. Add observations for each experiment based on the results")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()