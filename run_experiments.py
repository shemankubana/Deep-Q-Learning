"""
Hyperparameter Experiment Runner for Assault
Runs 10 different configurations as required by assignment
"""

import subprocess
import json
import os
from datetime import datetime

# 10 hyperparameter configurations for experiments
EXPERIMENTS = [
    {
        "id": 1,
        "name": "Baseline",
        "params": {
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Standard DQN - baseline for comparison"
    },
    {
        "id": 2,
        "name": "High Learning Rate",
        "params": {
            "lr": 5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Faster learning, may be less stable"
    },
    {
        "id": 3,
        "name": "Low Learning Rate",
        "params": {
            "lr": 5e-5,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Slower but more stable learning"
    },
    {
        "id": 4,
        "name": "High Gamma",
        "params": {
            "lr": 1e-4,
            "gamma": 0.995,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Values long-term rewards, better strategy"
    },
    {
        "id": 5,
        "name": "Low Gamma",
        "params": {
            "lr": 1e-4,
            "gamma": 0.95,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Focuses on immediate rewards"
    },
    {
        "id": 6,
        "name": "Large Batch",
        "params": {
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "More stable gradients"
    },
    {
        "id": 7,
        "name": "Small Batch",
        "params": {
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_size": 16,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Noisier but more frequent updates"
    },
    {
        "id": 8,
        "name": "Extended Exploration",
        "params": {
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.05,
            "exp_fraction": 0.2
        },
        "expected_behavior": "More exploration, may find better strategies"
    },
    {
        "id": 9,
        "name": "Quick Exploitation",
        "params": {
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.05
        },
        "expected_behavior": "Fast convergence, may miss optimal strategies"
    },
    {
        "id": 10,
        "name": "Aggressive",
        "params": {
            "lr": 1e-3,
            "gamma": 0.99,
            "batch_size": 64,
            "eps_start": 1.0,
            "eps_end": 0.02,
            "exp_fraction": 0.15
        },
        "expected_behavior": "Very fast learning with larger batches"
    }
]

def run_experiment(exp_config, env_name='AssaultNoFrameskip-v4', timesteps=500_000):
    """
    Run single experiment with specified configuration
    """
    exp_id = exp_config['id']
    params = exp_config['params']
    
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT {exp_id}: {exp_config['name']}")
    print(f"{'#'*80}")
    print(f"Expected: {exp_config['expected_behavior']}")
    print(f"\nHyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'#'*80}\n")
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--env', env_name,
        '--timesteps', str(timesteps),
        '--lr', str(params['lr']),
        '--gamma', str(params['gamma']),
        '--batch-size', str(params['batch_size']),
        '--eps-start', str(params['eps_start']),
        '--eps-end', str(params['eps_end']),
        '--exp-fraction', str(params['exp_fraction']),
        '--experiment', str(exp_id)
    ]
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Experiment {exp_id} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment {exp_id} failed: {e}\n")
        return False

def generate_results_table():
    """
    Generate markdown table for README
    """
    table = "## Hyperparameter Experiments\n\n"
    table += "| Exp | Name | lr | γ | Batch | ε Start | ε End | Exp Frac | Expected Behavior |\n"
    table += "|-----|------|-------|-------|-------|---------|---------|----------|-------------------|\n"
    
    for exp in EXPERIMENTS:
        p = exp['params']
        table += f"| {exp['id']} | {exp['name']} | {p['lr']} | {p['gamma']} | {p['batch_size']} | {p['eps_start']} | {p['eps_end']} | {p['exp_fraction']} | {exp['expected_behavior']} |\n"
    
    return table

def save_experiment_configs():
    """
    Save configurations to JSON
    """
    with open('experiment_configs.json', 'w') as f:
        json.dump(EXPERIMENTS, f, indent=2)
    print("✓ Configurations saved to experiment_configs.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter experiments on Assault')
    parser.add_argument('--env', type=str, default='AssaultNoFrameskip-v4',
                       help='Atari environment')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Training timesteps per experiment')
    parser.add_argument('--experiments', nargs='+', type=int,
                       help='Specific experiments to run (e.g., --experiments 1 2 3)')
    parser.add_argument('--generate-table', action='store_true',
                       help='Generate results table and exit')
    
    args = parser.parse_args()
    
    if args.generate_table:
        table = generate_results_table()
        print(table)
        with open('experiment_results_template.md', 'w') as f:
            f.write(table)
        print("\n✓ Table saved to experiment_results_template.md")
    else:
        save_experiment_configs()
        
        # Determine which experiments to run
        if args.experiments:
            experiments_to_run = [exp for exp in EXPERIMENTS if exp['id'] in args.experiments]
        else:
            experiments_to_run = EXPERIMENTS
        
        print(f"\n{'='*80}")
        print(f"Running {len(experiments_to_run)} experiments on Assault")
        print(f"Environment: {args.env}")
        print(f"Timesteps: {args.timesteps:,}")
        print(f"{'='*80}\n")
        
        # Run experiments
        results = {}
        for exp_config in experiments_to_run:
            success = run_experiment(exp_config, args.env, args.timesteps)
            results[exp_config['id']] = {
                'name': exp_config['name'],
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
        
        # Save results
        with open('experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        successful = sum(1 for r in results.values() if r['success'])
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {len(results) - successful}/{len(results)}")
        print(f"\nResults saved to experiment_results.json")
        print(f"{'='*80}\n")