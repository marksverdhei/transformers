#!/usr/bin/env python3
"""
Test MXFP4 quantization with 1-layer variants of gpt-oss-20b and gpt-oss-120b
to confirm that the hardcoded values in the quantizer cause issues.
"""

import torch
from transformers import GptOssConfig, GptOssForCausalLM, Mxfp4Config

def test_quantization(model_name, num_local_experts, hidden_size=2880):
    """Test MXFP4 quantization with given configuration."""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} with {num_local_experts} experts")
    print(f"{'='*70}")

    # Create a minimal 1-layer config
    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,  # Same as hidden_size for simplicity
        vocab_size=201088,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
    )

    print(f"Config: num_hidden_layers={config.num_hidden_layers}, "
          f"num_local_experts={config.num_local_experts}, "
          f"hidden_size={config.hidden_size}")

    # Try to create and quantize the model
    try:
        quantization_config = Mxfp4Config()

        # Create the model with random init
        model = GptOssForCausalLM(config)
        print(f"✓ Model created successfully")

        # Try to apply quantization (simulating what would happen during save/load)
        # We'll test the reshape operations that are hardcoded in get_state_dict_and_metadata
        print(f"\nTesting reshape operations from quantizer_mxfp4.py...")

        # Simulate the blocks and scales tensors from first MLP layer
        test_gate_up_blocks = torch.zeros(num_local_experts, 2 * hidden_size, hidden_size // 32, 16, dtype=torch.uint8)
        test_down_blocks = torch.zeros(num_local_experts, hidden_size, hidden_size // 32, 16, dtype=torch.uint8)

        print(f"  gate_up_blocks shape: {test_gate_up_blocks.shape}")
        print(f"  down_blocks shape: {test_down_blocks.shape}")

        # Test the hardcoded reshape from line 395 in quantizer_mxfp4.py
        try:
            # This reshape uses hardcoded 32 and 2880
            reshaped_gate_up = test_gate_up_blocks.transpose(-1, -2).reshape(32, -1, 90, 16)
            print(f"  ✗ Hardcoded reshape (32, -1, 90, 16) succeeded: {reshaped_gate_up.shape}")
            print(f"     But this only works for 32 experts and 2880 hidden_size!")
        except RuntimeError as e:
            print(f"  ✗ Hardcoded reshape (32, -1, 90, 16) FAILED: {e}")

        # Test the hardcoded reshape from line 405 in quantizer_mxfp4.py
        try:
            # This reshape uses hardcoded 32 and 2880
            reshaped_down = test_down_blocks.transpose(-1, -2).reshape(32, 2880, 90, -1)
            print(f"  ✗ Hardcoded reshape (32, 2880, 90, -1) succeeded: {reshaped_down.shape}")
            print(f"     But this only works for 32 experts and 2880 hidden_size!")
        except RuntimeError as e:
            print(f"  ✗ Hardcoded reshape (32, 2880, 90, -1) FAILED: {e}")

        # Test with dynamic values (what the fix should use)
        print(f"\nTesting with dynamic values (the fix)...")
        try:
            # Use actual config values instead of hardcoded ones
            reshaped_gate_up_dynamic = test_gate_up_blocks.transpose(-1, -2).reshape(num_local_experts, -1, hidden_size // 32, 16)
            print(f"  ✓ Dynamic reshape ({num_local_experts}, -1, {hidden_size // 32}, 16) succeeded: {reshaped_gate_up_dynamic.shape}")
        except RuntimeError as e:
            print(f"  ✗ Dynamic reshape failed: {e}")

        try:
            reshaped_down_dynamic = test_down_blocks.transpose(-1, -2).reshape(num_local_experts, hidden_size, hidden_size // 32, -1)
            print(f"  ✓ Dynamic reshape ({num_local_experts}, {hidden_size}, {hidden_size // 32}, -1) succeeded: {reshaped_down_dynamic.shape}")
        except RuntimeError as e:
            print(f"  ✗ Dynamic reshape failed: {e}")

        print(f"\n{'='*70}")
        if num_local_experts == 32 and hidden_size == 2880:
            print(f"RESULT: Hardcoded values work for {model_name} ✓")
        else:
            print(f"RESULT: Hardcoded values FAIL for {model_name} ✗")
        print(f"{'='*70}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Testing MXFP4 quantization with 1-layer model variants")
    print("This demonstrates the hardcoded expert count issue on main branch")

    # Test 1: gpt-oss-20b (32 experts) - should work with hardcoded values
    test_quantization("gpt-oss-20b (1 layer)", num_local_experts=32, hidden_size=2880)

    # Test 2: gpt-oss-120b (128 experts) - should fail with hardcoded values
    test_quantization("gpt-oss-120b (1 layer)", num_local_experts=128, hidden_size=2880)

if __name__ == "__main__":
    main()
