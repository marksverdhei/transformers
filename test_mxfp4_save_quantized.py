#!/usr/bin/env python3
"""
Test the actual save path for MXFP4 quantized models.
This tests the get_state_dict_and_metadata function that has hardcoded values.
"""

import tempfile
import torch
from transformers import GptOssConfig, GptOssForCausalLM, Mxfp4Config
import traceback

def test_save_quantized_model(model_name, num_local_experts, hidden_size=2880):
    """
    Test saving a quantized model (which triggers get_state_dict_and_metadata).
    This is where the hardcoded reshape happens on line 395 and 405.
    """
    print(f"\n{'='*70}")
    print(f"Testing save of quantized {model_name} with {num_local_experts} experts")
    print(f"{'='*70}")

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

    if not torch.cuda.is_available():
        print(f"   ⚠ No GPU available - skipping test")
        return

    try:
        # Step 1: Create and save a non-quantized model
        print("\n1. Creating non-quantized model...")
        model = GptOssForCausalLM(config)
        print(f"   ✓ Model created successfully")

        with tempfile.TemporaryDirectory() as tmpdir1:
            print(f"   Saving non-quantized model to: {tmpdir1}")
            model.save_pretrained(tmpdir1)
            print(f"   ✓ Non-quantized model saved")

            # Step 2: Load with quantization config (quantize on-the-fly)
            print("\n2. Loading with MXFP4 quantization config...")
            quantization_config = Mxfp4Config()

            try:
                quantized_model = GptOssForCausalLM.from_pretrained(
                    tmpdir1,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                print(f"   ✓ Model loaded and quantized successfully")

                # Step 3: Save the quantized model
                # THIS is where get_state_dict_and_metadata gets called with hardcoded values
                print("\n3. Saving quantized model (triggers get_state_dict_and_metadata)...")
                with tempfile.TemporaryDirectory() as tmpdir2:
                    print(f"   Saving to: {tmpdir2}")
                    quantized_model.save_pretrained(tmpdir2)
                    print(f"   ✓ Quantized model saved successfully")

                    # Try to load it back
                    print("\n4. Loading saved quantized model...")
                    reloaded_model = GptOssForCausalLM.from_pretrained(
                        tmpdir2,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                    print(f"   ✓ Model reloaded successfully")

                print(f"\n{'='*70}")
                print(f"RESULT: {model_name} full quantization cycle SUCCEEDED ✓")
                print(f"{'='*70}")

            except Exception as e:
                print(f"\n   ✗ Error during quantization/save cycle: {e}")
                print(f"\nFull traceback:")
                traceback.print_exc()
                print(f"\n{'='*70}")
                print(f"RESULT: {model_name} full quantization cycle FAILED ✗")
                print(f"{'='*70}")

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()


def main():
    print("="*70)
    print("Testing MXFP4 quantization with actual save of quantized models")
    print("This triggers get_state_dict_and_metadata with hardcoded reshape")
    print("="*70)

    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print(f"\n⚠ No GPU available - tests will be skipped")
        return

    # Test 1: gpt-oss-20b (32 experts) - should work with hardcoded values
    test_save_quantized_model("gpt-oss-20b (1 layer)", num_local_experts=32, hidden_size=2880)

    # Test 2: gpt-oss-120b (128 experts) - should FAIL with hardcoded values on main
    test_save_quantized_model("gpt-oss-120b (1 layer)", num_local_experts=128, hidden_size=2880)


if __name__ == "__main__":
    main()
