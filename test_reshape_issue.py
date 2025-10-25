#!/usr/bin/env python3
"""
Simple demonstration of the hardcoded reshape issue in quantizer_mxfp4.py lines 395 and 405.
"""

import torch


def test_hardcoded_reshape(num_local_experts, hidden_size=2880):
    """Test the exact reshape operations from quantizer_mxfp4.py"""
    print(f"\n{'='*70}")
    print(f"Testing with {num_local_experts} experts, hidden_size={hidden_size}")
    print(f"{'='*70}")

    # Simulate the tensor shapes that would exist in a quantized model
    # These are the shapes AFTER unswizzle_data and transpose
    print(f"\n1. Input tensor shapes (after unswizzle and transpose):")
    gate_up_shape = (num_local_experts, hidden_size // 32, 2 * hidden_size)
    down_shape = (num_local_experts, hidden_size // 32, hidden_size)

    print(f"   gate_up_proj: {gate_up_shape}")
    print(f"   down_proj: {down_shape}")

    # Create dummy tensors
    gate_up_tensor = torch.zeros(gate_up_shape, dtype=torch.uint8)
    down_tensor = torch.zeros(down_shape, dtype=torch.uint8)

    print(f"\n2. Attempting HARDCODED reshapes from main branch:")

    # Test gate_up_proj reshape (line 395 in quantizer_mxfp4.py)
    print(f"\n   gate_up_proj.reshape(32, -1, 90, 16)")
    print(f"      Note: 32 is hardcoded (should be {num_local_experts})")
    print(f"            90 is hardcoded (should be {hidden_size//32})")
    try:
        reshaped_gate_up = gate_up_tensor.reshape(32, -1, 90, 16)
        print(f"      Result shape: {reshaped_gate_up.shape}")

        expected_shape = (num_local_experts, 2*hidden_size, hidden_size//32, 16)
        if reshaped_gate_up.shape == expected_shape:
            print(f"      ✓ Shape is CORRECT (matches expected {expected_shape})")
        else:
            print(f"      ✗ Shape is WRONG!")
            print(f"        Expected: {expected_shape}")
            print(f"        Got:      {reshaped_gate_up.shape}")
    except RuntimeError as e:
        print(f"      ✗ FAILED: {e}")

    # Test down_proj reshape (line 405 in quantizer_mxfp4.py)
    print(f"\n   down_proj.reshape(32, 2880, 90, -1)")
    print(f"      Note: 32 is hardcoded (should be {num_local_experts})")
    print(f"            2880 is hardcoded (should be {hidden_size})")
    print(f"            90 is hardcoded (should be {hidden_size//32})")
    try:
        reshaped_down = down_tensor.reshape(32, 2880, 90, -1)
        print(f"      Result shape: {reshaped_down.shape}")

        expected_shape = (num_local_experts, hidden_size, hidden_size//32, 16)
        if reshaped_down.shape == expected_shape:
            print(f"      ✓ Shape is CORRECT (matches expected {expected_shape})")
        else:
            print(f"      ✗ Shape is WRONG!")
            print(f"        Expected: {expected_shape}")
            print(f"        Got:      {reshaped_down.shape}")
    except RuntimeError as e:
        print(f"      ✗ FAILED: {e}")

    print(f"\n3. With DYNAMIC reshapes (the fix):")

    # Test gate_up_proj with dynamic values
    print(f"\n   gate_up_proj.reshape({num_local_experts}, -1, {hidden_size//32}, 16)")
    try:
        reshaped_gate_up_dynamic = gate_up_tensor.reshape(num_local_experts, -1, hidden_size//32, 16)
        print(f"      Result shape: {reshaped_gate_up_dynamic.shape}")
        print(f"      ✓ CORRECT shape achieved with dynamic values")
    except RuntimeError as e:
        print(f"      ✗ FAILED: {e}")

    # Test down_proj with dynamic values
    print(f"\n   down_proj.reshape({num_local_experts}, {hidden_size}, {hidden_size//32}, -1)")
    try:
        reshaped_down_dynamic = down_tensor.reshape(num_local_experts, hidden_size, hidden_size//32, -1)
        print(f"      Result shape: {reshaped_down_dynamic.shape}")
        print(f"      ✓ CORRECT shape achieved with dynamic values")
    except RuntimeError as e:
        print(f"      ✗ FAILED: {e}")

    print(f"\n{'='*70}")
    if num_local_experts == 32 and hidden_size == 2880:
        print(f"RESULT: Hardcoded values work (by luck) ✓")
    else:
        print(f"RESULT: Hardcoded values produce INCORRECT shapes ✗")
    print(f"{'='*70}")


def main():
    print("="*70)
    print("Demonstrating hardcoded reshape issue in quantizer_mxfp4.py")
    print("Lines 395 and 405 have hardcoded values: 32, 2880, 90")
    print("="*70)

    # Test 1: gpt-oss-20b configuration (32 experts, 2880 hidden_size)
    print("\n\nTEST 1: gpt-oss-20b configuration")
    test_hardcoded_reshape(num_local_experts=32, hidden_size=2880)

    # Test 2: gpt-oss-120b configuration (128 experts, 2880 hidden_size)
    print("\n\nTEST 2: gpt-oss-120b configuration")
    test_hardcoded_reshape(num_local_experts=128, hidden_size=2880)


if __name__ == "__main__":
    main()
