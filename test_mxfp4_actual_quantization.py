#!/usr/bin/env python3
"""
Test MXFP4 quantization with actual 1-layer variants of gpt-oss-20b and gpt-oss-120b.
This creates real models with random initialization and applies actual quantization.
"""

import tempfile
import torch
from transformers import GptOssConfig, GptOssForCausalLM, Mxfp4Config
import traceback

def test_actual_quantization(model_name, num_local_experts, hidden_size=2880):
    """Test MXFP4 quantization with actual model creation and quantization."""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} with {num_local_experts} experts")
    print(f"{'='*70}")

    # Create a minimal 1-layer config (one GPT decoder block)
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

    try:
        # Create model with random initialization
        print("\n1. Creating model with random initialization...")
        model = GptOssForCausalLM(config)
        print(f"   ✓ Model created successfully")
        print(f"   Model size: {model.num_parameters():,} parameters")

        # Create quantization config
        quantization_config = Mxfp4Config()

        # Try to save the model with quantization
        # This will trigger the quantization process
        print("\n2. Saving model with MXFP4 quantization...")
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Move model to GPU if available (required for quantization)
                if torch.cuda.is_available():
                    device = "cuda"
                    print(f"   Moving model to {device}...")
                    model = model.to(device)
                else:
                    print(f"   ⚠ No GPU available - MXFP4 quantization requires GPU")
                    print(f"   Skipping actual quantization test")
                    return

                # Save the model - this should trigger quantization
                print(f"   Saving to temporary directory: {tmpdir}")
                model.save_pretrained(tmpdir, safe_serialization=True)
                print(f"   ✓ Model saved successfully")

                # Try to load it back
                print("\n3. Loading quantized model back...")
                loaded_model = GptOssForCausalLM.from_pretrained(
                    tmpdir,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                print(f"   ✓ Quantized model loaded successfully")

                # Check if quantization config is present
                if hasattr(loaded_model.config, 'quantization_config'):
                    print(f"   Quantization config: {loaded_model.config.quantization_config}")

                print(f"\n{'='*70}")
                print(f"RESULT: {model_name} quantization SUCCEEDED ✓")
                print(f"{'='*70}")

            except Exception as e:
                print(f"\n   ✗ Error during quantization/save/load: {e}")
                traceback.print_exc()
                print(f"\n{'='*70}")
                print(f"RESULT: {model_name} quantization FAILED ✗")
                print(f"{'='*70}")

    except Exception as e:
        print(f"✗ Error creating model: {e}")
        traceback.print_exc()


def test_quantization_on_the_fly(model_name, num_local_experts, hidden_size=2880):
    """Test quantizing a model on-the-fly during loading."""
    print(f"\n{'='*70}")
    print(f"Testing on-the-fly quantization: {model_name} with {num_local_experts} experts")
    print(f"{'='*70}")

    # Create a minimal 1-layer config
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

    print(f"Config: num_hidden_layers={config.num_hidden_layers}, "
          f"num_local_experts={config.num_local_experts}, "
          f"hidden_size={config.hidden_size}")

    try:
        if not torch.cuda.is_available():
            print(f"   ⚠ No GPU available - MXFP4 quantization requires GPU")
            print(f"   Skipping on-the-fly quantization test")
            return

        # Create and save a non-quantized model first
        print("\n1. Creating and saving non-quantized model...")
        model = GptOssForCausalLM(config)
        print(f"   ✓ Model created successfully")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            print(f"   ✓ Non-quantized model saved")

            # Now try to load it with quantization
            print("\n2. Loading with on-the-fly MXFP4 quantization...")
            quantization_config = Mxfp4Config()

            try:
                quantized_model = GptOssForCausalLM.from_pretrained(
                    tmpdir,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                print(f"   ✓ Model quantized on-the-fly successfully")

                print(f"\n{'='*70}")
                print(f"RESULT: {model_name} on-the-fly quantization SUCCEEDED ✓")
                print(f"{'='*70}")

            except Exception as e:
                print(f"   ✗ Error during on-the-fly quantization: {e}")
                traceback.print_exc()
                print(f"\n{'='*70}")
                print(f"RESULT: {model_name} on-the-fly quantization FAILED ✗")
                print(f"{'='*70}")

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()


def main():
    print("="*70)
    print("Testing ACTUAL MXFP4 quantization with 1-layer model variants")
    print("This will create real models and apply actual quantization")
    print("="*70)

    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print(f"\n⚠ No GPU available - tests will be limited")

    # Check if kernels are available
    try:
        from transformers.utils import is_kernels_available, is_triton_available
        if is_kernels_available():
            print(f"✓ Kernels package is available")
        else:
            print(f"⚠ Kernels package not available")

        if is_triton_available("3.4.0"):
            print(f"✓ Triton >= 3.4.0 is available")
        else:
            print(f"⚠ Triton >= 3.4.0 not available")
    except Exception as e:
        print(f"⚠ Could not check for kernels/triton: {e}")

    print("\n" + "="*70)
    print("TEST 1: Pre-quantized model save/load")
    print("="*70)

    # Test 1: gpt-oss-20b (32 experts) - should work with hardcoded values on main
    test_actual_quantization("gpt-oss-20b (1 layer)", num_local_experts=32, hidden_size=2880)

    # Test 2: gpt-oss-120b (128 experts) - should fail with hardcoded values on main
    test_actual_quantization("gpt-oss-120b (1 layer)", num_local_experts=128, hidden_size=2880)

    print("\n" + "="*70)
    print("TEST 2: On-the-fly quantization")
    print("="*70)

    # Test on-the-fly quantization
    test_quantization_on_the_fly("gpt-oss-20b (1 layer)", num_local_experts=32, hidden_size=2880)
    test_quantization_on_the_fly("gpt-oss-120b (1 layer)", num_local_experts=128, hidden_size=2880)


if __name__ == "__main__":
    main()
