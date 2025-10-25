#!/usr/bin/env python3
"""
Direct test of the get_state_dict_and_metadata function in quantizer_mxfp4.py
This function has hardcoded values on lines 395 and 405.
"""

import torch
from transformers import GptOssConfig
from transformers.integrations.mxfp4 import Mxfp4GptOssExperts
from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
from transformers import Mxfp4Config
import traceback


def create_mock_quantized_model(num_local_experts, hidden_size=2880):
    """Create a minimal mock model with Mxfp4GptOssExperts to test get_state_dict_and_metadata"""

    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        vocab_size=201088,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
    )

    # Create a minimal model-like object
    class MockModel:
        def __init__(self, config):
            self.config = config
            # Create a mock expert layer
            self.experts = Mxfp4GptOssExperts(config)

        def named_modules(self):
            yield ("model.layers.0.mlp.experts", self.experts)

        def state_dict(self):
            # Return a minimal state dict
            return {}

    return MockModel(config)


def test_get_state_dict_hardcoded_reshape(model_name, num_local_experts, hidden_size=2880):
    """Test the get_state_dict_and_metadata function with hardcoded reshapes"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} with {num_local_experts} experts")
    print(f"{'='*70}")

    print(f"Config: num_local_experts={num_local_experts}, hidden_size={hidden_size}")

    try:
        # Create mock model
        print("\n1. Creating mock quantized model...")
        model = create_mock_quantized_model(num_local_experts, hidden_size)
        print(f"   ✓ Mock model created")

        # Create quantizer
        quantization_config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(quantization_config)

        # Simulate quantized tensors (the experts module needs these attributes)
        # In a real quantized model, these would be set during loading
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("   ⚠ No GPU, using CPU")
            device = torch.device("cpu")

        # Create mock quantized data structures
        # These simulate what would be in a quantized model
        class MockTensor:
            def __init__(self, data):
                self.storage = type('obj', (object,), {
                    'data': data,
                    'layout': type('obj', (object,), {
                        'unswizzle_data': lambda x: x.transpose(-1, -2)
                    })()
                })()

        class MockPrecisionConfig:
            def __init__(self, scales):
                self.weight_scale = MockTensor(scales)

        # Set up the mock quantized tensors with correct shapes for each case
        print(f"\n2. Setting up mock quantized tensors...")
        print(f"   Creating gate_up_proj_blocks: ({num_local_experts}, {2*hidden_size}, {hidden_size//32}, 16)")
        print(f"   Creating down_proj_blocks: ({num_local_experts}, {hidden_size}, {hidden_size//32}, 16)")

        gate_up_data = torch.zeros(16, hidden_size//32, 2*hidden_size, num_local_experts, dtype=torch.uint8, device=device)
        gate_up_scales = torch.zeros(hidden_size//32, 2*hidden_size, num_local_experts, dtype=torch.uint8, device=device)
        down_data = torch.zeros(16, hidden_size//32, hidden_size, num_local_experts, dtype=torch.uint8, device=device)
        down_scales = torch.zeros(hidden_size//32, hidden_size, num_local_experts, dtype=torch.uint8, device=device)

        model.experts.gate_up_proj = MockTensor(gate_up_data)
        model.experts.gate_up_proj_precision_config = MockPrecisionConfig(gate_up_scales)
        model.experts.down_proj = MockTensor(down_data)
        model.experts.down_proj_precision_config = MockPrecisionConfig(down_scales)

        print(f"   ✓ Mock tensors created")

        # Now call get_state_dict_and_metadata which has the hardcoded reshapes
        print(f"\n3. Calling get_state_dict_and_metadata (has hardcoded reshapes)...")
        print(f"   This will attempt:")
        print(f"   - Line 395: .reshape(32, -1, 90, 16)  # hardcoded 32 experts!")
        print(f"   - Line 405: .reshape(32, 2880, 90, -1)  # hardcoded 32 experts, 2880 hidden_size!")

        try:
            state_dict, metadata = quantizer.get_state_dict_and_metadata(model, safe_serialization=True)
            print(f"\n   ✗ get_state_dict_and_metadata succeeded!")
            print(f"      But notice the tensor shapes in state_dict:")
            for key, tensor in state_dict.items():
                if 'blocks' in key or 'scales' in key:
                    print(f"      {key}: {tensor.shape}")

            # Check if shapes are correct
            expected_gate_up_shape = (num_local_experts, 2*hidden_size, hidden_size//32, 16)
            expected_down_shape = (num_local_experts, hidden_size, hidden_size//32, 16)

            gate_up_key = "model.layers.0.mlp.experts.gate_up_proj_blocks"
            down_key = "model.layers.0.mlp.experts.down_proj_blocks"

            if gate_up_key in state_dict:
                actual_shape = tuple(state_dict[gate_up_key].shape)
                if actual_shape == expected_gate_up_shape:
                    print(f"\n   ✓ gate_up_proj_blocks has CORRECT shape: {actual_shape}")
                else:
                    print(f"\n   ✗ gate_up_proj_blocks has WRONG shape!")
                    print(f"      Expected: {expected_gate_up_shape}")
                    print(f"      Got:      {actual_shape}")

            if down_key in state_dict:
                actual_shape = tuple(state_dict[down_key].shape)
                if actual_shape == expected_down_shape:
                    print(f"   ✓ down_proj_blocks has CORRECT shape: {actual_shape}")
                else:
                    print(f"   ✗ down_proj_blocks has WRONG shape!")
                    print(f"      Expected: {expected_down_shape}")
                    print(f"      Got:      {actual_shape}")

            print(f"\n{'='*70}")
            if num_local_experts == 32 and hidden_size == 2880:
                print(f"RESULT: {model_name} - Hardcoded values produce correct shapes ✓")
            else:
                print(f"RESULT: {model_name} - Hardcoded values produce WRONG shapes ✗")
            print(f"{'='*70}")

        except Exception as e:
            print(f"\n   ✗ get_state_dict_and_metadata FAILED with error:")
            print(f"      {e}")
            print(f"\n   Full traceback:")
            traceback.print_exc()
            print(f"\n{'='*70}")
            print(f"RESULT: {model_name} - FAILED ✗")
            print(f"{'='*70}")

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()


def main():
    print("="*70)
    print("Direct test of get_state_dict_and_metadata hardcoded reshapes")
    print("Lines 395 and 405 in quantizer_mxfp4.py")
    print("="*70)

    # Test 1: gpt-oss-20b (32 experts) - hardcoded values match
    test_get_state_dict_hardcoded_reshape(
        "gpt-oss-20b (1 layer)",
        num_local_experts=32,
        hidden_size=2880
    )

    # Test 2: gpt-oss-120b (128 experts) - hardcoded values DON'T match
    test_get_state_dict_hardcoded_reshape(
        "gpt-oss-120b (1 layer)",
        num_local_experts=128,
        hidden_size=2880
    )


if __name__ == "__main__":
    main()
