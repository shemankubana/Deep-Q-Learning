#!/usr/bin/env python3
"""
Check and fix Atari environment setup
"""

import sys

def check_atari_setup():
    """Check if Atari environments are available"""
    
    print("="*60)
    print("Checking Atari Environment Setup")
    print("="*60)
    print()
    
    # Check imports
    print("1. Checking imports...")
    try:
        import gymnasium as gym
        print(f"   ✓ gymnasium {gym.__version__}")
    except ImportError as e:
        print(f"   ✗ gymnasium not installed: {e}")
        return False
    
    try:
        import ale_py
        print(f"   ✓ ale_py {ale_py.__version__}")
    except ImportError as e:
        print(f"   ✗ ale_py not installed: {e}")
        return False
    
    # Register ALE environments
    print()
    print("2. Registering ALE environments...")
    try:
        from ale_py import ALEInterface
        gym.register_envs(ale_py)
        print("   ✓ ALE environments registered")
    except Exception as e:
        print(f"   ⚠ Warning: {e}")
    
    # Check for Atari environments
    print()
    print("3. Checking available Atari environments...")
    
    all_envs = gym.envs.registry.keys()
    atari_envs = [env for env in all_envs if 'NoFrameskip' in env or 'ALE/' in env]
    
    if atari_envs:
        print(f"   ✓ Found {len(atari_envs)} Atari environments")
        print()
        print("   Available Atari games (first 10):")
        for env in list(atari_envs)[:10]:
            print(f"     - {env}")
        if len(atari_envs) > 10:
            print(f"     ... and {len(atari_envs) - 10} more")
    else:
        print("   ✗ No Atari environments found!")
        return False
    
    # Test creating an environment
    print()
    print("4. Testing environment creation...")
    
    test_envs = [
        'ALE/Breakout-v5',
        'BreakoutNoFrameskip-v4',
        'Breakout-v4'
    ]
    
    success = False
    working_env = None
    
    for env_name in test_envs:
        try:
            env = gym.make(env_name)
            print(f"   ✓ Successfully created: {env_name}")
            env.close()
            working_env = env_name
            success = True
            break
        except Exception as e:
            print(f"   ✗ Failed to create {env_name}: {type(e).__name__}")
    
    print()
    print("="*60)
    if success:
        print("✓ ATARI SETUP SUCCESSFUL!")
        print("="*60)
        print()
        print(f"Use this environment name: {working_env}")
        print()
        print("Example command:")
        print(f"  python train.py --env {working_env} --timesteps 1000")
        return True
    else:
        print("✗ ATARI SETUP FAILED!")
        print("="*60)
        print()
        print("Fix: Run these commands:")
        print("  pip install gymnasium[atari,accept-rom-license]")
        print("  pip install ale-py")
        print()
        print("Or try:")
        print("  pip install 'gymnasium[atari]' autorom[accept-rom-license]")
        print("  AutoROM --accept-license")
        return False

if __name__ == "__main__":
    success = check_atari_setup()
    sys.exit(0 if success else 1)