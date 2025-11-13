#!/usr/bin/env python3
"""
Environment Setup Verification Script
Checks if all dependencies are properly installed for Assault DQN training
"""

import sys

def check_setup():
    """Verify all requirements are installed"""
    
    print("="*60)
    print("CHECKING ENVIRONMENT SETUP FOR ASSAULT DQN")
    print("="*60)
    print()
    
    all_good = True
    
    # Check Python version
    print("1. Python Version")
    print("-" * 40)
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro}")
        print("   Need Python 3.8 or higher")
        all_good = False
    print()
    
    # Check imports
    print("2. Required Packages")
    print("-" * 40)
    
    packages = [
        ("gymnasium", "gym"),
        ("ale_py", "ale_py"),
        ("stable_baselines3", "stable_baselines3"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("tensorboard", "tensorboard"),
        ("moviepy", "moviepy.editor"),
    ]
    
    for name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"   ✓ {name} ({version})")
        except ImportError:
            print(f"   ✗ {name} not installed")
            all_good = False
    print()
    
    # Check Gymnasium Atari
    print("3. Gymnasium Atari Environments")
    print("-" * 40)
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        print("   ✓ ALE environments registered")
    except Exception as e:
        print(f"   ✗ Failed to register ALE: {e}")
        all_good = False
    print()
    
    # Check Assault environment
    print("4. Assault Environment")
    print("-" * 40)
    test_envs = ['ALE/Assault-v5', 'AssaultNoFrameskip-v4']
    
    working_env = None
    for env_name in test_envs:
        try:
            env = gym.make(env_name)
            print(f"   ✓ {env_name} available")
            env.close()
            if working_env is None:
                working_env = env_name
        except Exception as e:
            print(f"   ✗ {env_name}: {type(e).__name__}")
    
    if working_env:
        print(f"\n   Recommended: {working_env}")
    else:
        print("\n   ✗ No Assault environment available")
        all_good = False
    print()
    
    # Test environment creation
    print("5. Environment Test")
    print("-" * 40)
    if working_env:
        try:
            env = gym.make(working_env)
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   ✓ Environment works correctly")
            print(f"   Action space: {env.action_space}")
            print(f"   Observation shape: {obs.shape}")
            env.close()
        except Exception as e:
            print(f"   ✗ Environment test failed: {e}")
            all_good = False
    print()
    
    # Check GPU availability
    print("6. GPU Availability (Optional)")
    print("-" * 40)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("   ⚠ No GPU detected (will use CPU)")
            print("   Training will be slower but will work")
    except Exception as e:
        print(f"   ⚠ Could not check GPU: {e}")
    print()
    
    # Final verdict
    print("="*60)
    if all_good:
        print("✓ SETUP COMPLETE - Ready to train!")
        print("="*60)
        print()
        print("Next steps:")
        print("  1. Quick test: python train.py --timesteps 10000")
        print("  2. Full training: python train.py --timesteps 500000")
        print("  3. Play: python play.py --episodes 5")
        print()
        return True
    else:
        print("✗ SETUP INCOMPLETE - Please fix the issues above")
        print("="*60)
        print()
        print("To fix:")
        print("  pip install -r requirements.txt")
        print()
        return False

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)