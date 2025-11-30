#!/usr/bin/env python
"""
Quick test script to verify PPO implementation can be imported and initialized.
This doesn't run actual training, just checks that the code is syntactically correct.
"""

import sys
import os
import pathlib

# Add parent directory to path
sys.path.insert(0, os.path.join(pathlib.Path(__file__).parent.parent.absolute()))

def test_import():
    """Test that PPO agent can be imported"""
    print("Testing PPO agent import...")
    try:
        from alfworld.agents.agent.text_ppo_agent import TextPPOAgent, PPOMemory
        print("✓ Successfully imported TextPPOAgent and PPOMemory")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_ppo_memory():
    """Test PPO memory operations"""
    print("\nTesting PPO memory...")
    try:
        from alfworld.agents.agent.text_ppo_agent import PPOMemory

        memory = PPOMemory()
        print(f"✓ Created PPOMemory instance")

        # Test push
        memory.push(
            observation="test obs",
            task_desc="test task",
            action="go north",
            action_idx=0,
            action_candidate_list=["go north", "go south"],
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-0.1,
            dynamic=None
        )
        print(f"✓ Pushed sample to memory, size: {len(memory)}")

        # Test get_batches
        batches = memory.get_batches()
        print(f"✓ Retrieved batches, keys: {list(batches.keys())}")

        # Test reset
        memory.reset()
        print(f"✓ Reset memory, size: {len(memory)}")

        return True
    except Exception as e:
        print(f"✗ PPO memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test that config file exists and is valid YAML"""
    print("\nTesting PPO config file...")
    try:
        import yaml
        config_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'ppo_config.yaml'
        )

        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✓ Config file loaded successfully")

        # Check for required PPO sections
        if 'ppo' not in config:
            print("✗ Missing 'ppo' section in config")
            return False

        ppo_config = config['ppo']
        required_keys = [
            'clip_epsilon', 'ppo_epochs', 'minibatch_size',
            'value_loss_coef', 'entropy_coef', 'max_grad_norm',
            'gae_lambda', 'discount_gamma'
        ]

        missing_keys = [key for key in required_keys if key not in ppo_config]
        if missing_keys:
            print(f"✗ Missing PPO config keys: {missing_keys}")
            return False

        print(f"✓ All required PPO config keys present")
        print(f"  - clip_epsilon: {ppo_config['clip_epsilon']}")
        print(f"  - ppo_epochs: {ppo_config['ppo_epochs']}")
        print(f"  - discount_gamma: {ppo_config['discount_gamma']}")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_init():
    """Test that agent can be initialized (requires full setup)"""
    print("\nTesting PPO agent initialization...")
    print("Note: This test requires ALFWORLD_DATA to be set and may take a while...")

    # Check if ALFWORLD_DATA is set
    if 'ALFWORLD_DATA' not in os.environ:
        print("⚠ ALFWORLD_DATA not set, skipping agent initialization test")
        print("  To run this test, set: export ALFWORLD_DATA=/path/to/alfworld/data")
        return None

    try:
        import alfworld.agents.modules.generic as generic
        from alfworld.agents.agent.text_ppo_agent import TextPPOAgent

        # Load config
        config_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'ppo_config.yaml'
        )
        sys.argv = [sys.argv[0], config_path]

        print("  Loading config...")
        config = generic.load_config()

        print("  Initializing agent (this may take a minute)...")
        agent = TextPPOAgent(config)

        print(f"✓ Successfully initialized TextPPOAgent")
        print(f"  - Training method: {agent.training_method}")
        print(f"  - Batch size: {agent.batch_size}")
        print(f"  - Block hidden dim: {agent.online_net.block_hidden_dim}")

        # Check PPO-specific attributes
        assert hasattr(agent, 'ppo_memory'), "Missing ppo_memory"
        assert hasattr(agent, 'value_head'), "Missing value_head"
        assert hasattr(agent, 'clip_epsilon'), "Missing clip_epsilon"
        print(f"  - Clip epsilon: {agent.clip_epsilon}")
        print(f"  - PPO epochs: {agent.ppo_epochs}")

        return True
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("PPO Implementation Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Import Test", test_import()))
    results.append(("PPO Memory Test", test_ppo_memory()))
    results.append(("Config File Test", test_config()))
    results.append(("Agent Init Test", test_agent_init()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)

    for test_name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{status:8s} - {test_name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n⚠ Some tests failed. Please fix the issues before training.")
        sys.exit(1)
    elif skipped > 0:
        print("\n⚠ Some tests were skipped. The implementation may work but needs full testing.")
        sys.exit(0)
    else:
        print("\n✓ All tests passed! PPO implementation is ready to use.")
        sys.exit(0)

if __name__ == '__main__':
    main()
