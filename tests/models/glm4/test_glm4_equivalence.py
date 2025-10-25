# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Equivalence investigation between GLM4 (dense) and GLM4-MoE.

NOTE: GLM4 and GLM4-MoE have fundamentally different architectures:

MLP (Mathematically Equivalent):
- GLM4 uses Phi3MLP (fused gate_up_proj)
- GLM4-MoE uses DeepseekV3MLP (separate gate_proj/up_proj)
- These are mathematically equivalent - weights can be converted between formats

Layer Normalization (NOT Equivalent - PRIMARY BLOCKER):
- GLM4: 4 layer norms per layer (pre+post norm architecture)
- GLM4-MoE: 2 layer norms per layer (standard pre-norm architecture)
- This is a fundamental computation graph difference

These tests investigate whether GLM4-MoE configured with minimal MoE can approximate GLM4
behavior when weights are properly converted, documenting the architectural barriers.
"""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import (
        Glm4Config,
        Glm4ForCausalLM,
        Glm4MoeConfig,
        Glm4MoeForCausalLM,
    )


@require_torch
class Glm4MoeMinimalExpertTest(unittest.TestCase):
    """Test GLM4-MoE with minimal expert configuration (1 routed + 1 shared)."""

    def test_glm4_moe_with_minimal_moe_configuration(self):
        """Verify GLM4-MoE with minimal MoE setup (2 routed experts + 1 shared)."""
        torch.manual_seed(42)

        config = Glm4MoeConfig(
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,  # Size for routed experts
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
            first_k_dense_replace=0,  # All layers use MoE
            n_routed_experts=2,  # Minimum 2 experts (routing requires >=2 per group)
            n_shared_experts=1,  # Single shared expert
            num_experts_per_tok=1,  # Route to 1 expert per token
            n_group=1,  # Single group
            topk_group=1,
            pad_token_id=0,
            eos_token_id=1,
        )

        model = Glm4MoeForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 10), device=torch_device)

        with torch.no_grad():
            output = model(input_ids)

        # Verify output shape
        self.assertEqual(output.logits.shape, (2, 10, 1000))

        # Verify no NaN
        self.assertFalse(torch.isnan(output.logits).any())

    def test_glm4_moe_all_dense_layers(self):
        """Verify GLM4-MoE can use all-dense configuration via first_k_dense_replace."""
        torch.manual_seed(42)

        config = Glm4MoeConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
            first_k_dense_replace=4,  # All 4 layers are dense (no MoE)
            n_routed_experts=1,  # Ignored when all layers are dense
            n_shared_experts=0,
            pad_token_id=0,
            eos_token_id=1,
        )

        model = Glm4MoeForCausalLM(config).to(torch_device)
        model.eval()

        # Verify layers are dense
        for i, layer in enumerate(model.model.layers):
            mlp_class_name = layer.mlp.__class__.__name__
            # Should be Glm4MoeMLP when dense (inherits from DeepseekV3MLP -> Qwen2MoeMLP -> GemmaMLP)
            self.assertIn("MLP", mlp_class_name)
            # Should NOT have MoE components
            self.assertFalse(hasattr(layer.mlp, 'gate'), f"Layer {i} should not have MoE gate")

        input_ids = torch.randint(0, 1000, (2, 10), device=torch_device)

        with torch.no_grad():
            output = model(input_ids)

        self.assertEqual(output.logits.shape, (2, 10, 1000))
        self.assertFalse(torch.isnan(output.logits).any())

    def test_architectural_differences_documented(self):
        """
        Document the architectural differences between GLM4 and GLM4-MoE.

        This test serves as documentation that these models have different architectures
        and cannot be directly compared for exact equivalence.
        """
        torch.manual_seed(42)

        glm4_config = Glm4Config(
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            head_dim=32,
            pad_token_id=0,
        )

        glm4_moe_config = Glm4MoeConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            first_k_dense_replace=1,
            pad_token_id=0,
        )

        glm4_model = Glm4ForCausalLM(glm4_config).to(torch_device)
        glm4_moe_model = Glm4MoeForCausalLM(glm4_moe_config).to(torch_device)

        # Document MLP differences
        glm4_mlp = glm4_model.model.layers[0].mlp
        glm4_moe_mlp = glm4_moe_model.model.layers[0].mlp

        # GLM4 uses Phi3MLP (fused)
        self.assertTrue(hasattr(glm4_mlp, 'gate_up_proj'))
        self.assertFalse(hasattr(glm4_mlp, 'gate_proj'))

        # GLM4-MoE uses Glm4MoeMLP (inherits from DeepseekV3MLP, separate projections)
        self.assertTrue(hasattr(glm4_moe_mlp, 'gate_proj'))
        self.assertTrue(hasattr(glm4_moe_mlp, 'up_proj'))
        self.assertFalse(hasattr(glm4_moe_mlp, 'gate_up_proj'))

        # Document layer norm differences
        glm4_layer = glm4_model.model.layers[0]
        glm4_moe_layer = glm4_moe_model.model.layers[0]

        # GLM4 has 4 layer norms
        self.assertTrue(hasattr(glm4_layer, 'input_layernorm'))
        self.assertTrue(hasattr(glm4_layer, 'post_attention_layernorm'))
        self.assertTrue(hasattr(glm4_layer, 'post_self_attn_layernorm'))
        self.assertTrue(hasattr(glm4_layer, 'post_mlp_layernorm'))

        # GLM4-MoE has 2 layer norms (from DeepseekV3DecoderLayer)
        self.assertTrue(hasattr(glm4_moe_layer, 'input_layernorm'))
        self.assertTrue(hasattr(glm4_moe_layer, 'post_attention_layernorm'))
        self.assertFalse(hasattr(glm4_moe_layer, 'post_self_attn_layernorm'))
        self.assertFalse(hasattr(glm4_moe_layer, 'post_mlp_layernorm'))

        # Document attention differences
        glm4_attn_class = glm4_model.model.layers[0].self_attn.__class__.__name__
        glm4_moe_attn_class = glm4_moe_model.model.layers[0].self_attn.__class__.__name__

        # Different attention implementations
        self.assertEqual(glm4_attn_class, "Glm4Attention")
        self.assertEqual(glm4_moe_attn_class, "Glm4MoeAttention")

        print("\n" + "="*70)
        print("GLM4 vs GLM4-MoE Architectural Differences:")
        print("="*70)
        print(f"\nMLP (Mathematically Equivalent ✅):")
        print(f"  GLM4:     {glm4_mlp.__class__.__name__} (fused gate_up_proj)")
        print(f"  GLM4-MoE: {glm4_moe_mlp.__class__.__name__} (separate gate_proj/up_proj)")
        print(f"  → These compute the same function, just different weight layout")
        print(f"  → Weights can be converted by splitting/concatenating")
        print(f"\nLayer Norms (NOT Equivalent ❌ - PRIMARY BLOCKER):")
        print(f"  GLM4:     4 norms (input, post_attention, post_self_attn, post_mlp)")
        print(f"  GLM4-MoE: 2 norms (input, post_attention)")
        print(f"  → GLM4 uses pre+post norm, GLM4-MoE uses standard pre-norm")
        print(f"  → Different computation graph structure")
        print(f"\nAttention:")
        print(f"  GLM4:     {glm4_attn_class}")
        print(f"  GLM4-MoE: {glm4_moe_attn_class}")
        print("\nConclusion: Models CANNOT be equivalent due to layer norm differences.")
        print("The MLP differences are superficial (weight layout only), but the")
        print("layer normalization architecture is fundamentally different.")
        print("="*70 + "\n")
