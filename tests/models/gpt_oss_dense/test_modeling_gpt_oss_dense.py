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
"""Testing suite for the PyTorch GptOssDense model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        GptOssDenseConfig,
        GptOssDenseForCausalLM,
        GptOssDenseModel,
        GptOssConfig,
        GptOssForCausalLM,
    )


class GptOssDenseModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GptOssDenseModel


@require_torch
class GptOssDenseModelTest(CausalLMModelTest, unittest.TestCase):
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = GptOssDenseModelTester

    @unittest.skip("GptOssDense forcefully disables sdpa due to Sink")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("GptOssDense eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip("GptOssDense eager/FA2 attention outputs are expected to be different")
    def test_flash_attn_2_equivalence(self):
        pass

    @unittest.skip("Padding handling may differ")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("GptOssDense does not support flex officially")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skipIf(torch_device == "cpu", "GptOssDense does not support flex officially")
    def test_generate_compile_model_forward_fullgraph(self):
        return super().test_generate_compile_model_forward_fullgraph()


@require_torch
class GptOssDenseEquivalenceTest(unittest.TestCase):
    """Test that GptOssDense produces equivalent outputs to GptOss with 1 expert."""

    def _create_models_with_shared_weights(self, dtype=torch.float32):
        """Helper to create MoE and Dense models with shared weights."""
        torch.manual_seed(42)

        moe_config = GptOssConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            num_local_experts=1,
            num_experts_per_tok=1,
            swiglu_limit=7.0,
            max_position_embeddings=512,
        )

        dense_config = GptOssDenseConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            swiglu_limit=7.0,
            max_position_embeddings=512,
        )

        moe_model = GptOssForCausalLM(moe_config).to(torch_device).to(dtype)
        dense_model = GptOssDenseForCausalLM(dense_config).to(torch_device).to(dtype)

        moe_model.eval()
        dense_model.eval()

        # Copy weights from MoE to Dense
        with torch.no_grad():
            dense_model.model.embed_tokens.weight.copy_(moe_model.model.embed_tokens.weight)
            dense_model.model.norm.weight.copy_(moe_model.model.norm.weight)
            dense_model.lm_head.weight.copy_(moe_model.lm_head.weight)

            for dense_layer, moe_layer in zip(dense_model.model.layers, moe_model.model.layers):
                # Attention
                dense_layer.self_attn.q_proj.weight.copy_(moe_layer.self_attn.q_proj.weight)
                dense_layer.self_attn.q_proj.bias.copy_(moe_layer.self_attn.q_proj.bias)
                dense_layer.self_attn.k_proj.weight.copy_(moe_layer.self_attn.k_proj.weight)
                dense_layer.self_attn.k_proj.bias.copy_(moe_layer.self_attn.k_proj.bias)
                dense_layer.self_attn.v_proj.weight.copy_(moe_layer.self_attn.v_proj.weight)
                dense_layer.self_attn.v_proj.bias.copy_(moe_layer.self_attn.v_proj.bias)
                dense_layer.self_attn.o_proj.weight.copy_(moe_layer.self_attn.o_proj.weight)
                dense_layer.self_attn.o_proj.bias.copy_(moe_layer.self_attn.o_proj.bias)
                dense_layer.self_attn.sinks.copy_(moe_layer.self_attn.sinks)

                # Layer norms
                dense_layer.input_layernorm.weight.copy_(moe_layer.input_layernorm.weight)
                dense_layer.post_attention_layernorm.weight.copy_(moe_layer.post_attention_layernorm.weight)

                # MLP: copy expert 0 weights with transpose
                dense_layer.mlp.gate_up_proj.weight.copy_(moe_layer.mlp.experts.gate_up_proj[0].T)
                dense_layer.mlp.gate_up_proj.bias.copy_(moe_layer.mlp.experts.gate_up_proj_bias[0])
                dense_layer.mlp.down_proj.weight.copy_(moe_layer.mlp.experts.down_proj[0].T)
                dense_layer.mlp.down_proj.bias.copy_(moe_layer.mlp.experts.down_proj_bias[0])

        return moe_model, dense_model

    def test_equivalence_with_single_expert_moe(self):
        """Dense model should match MoE with 1 expert when weights are copied."""
        moe_model, dense_model = self._create_models_with_shared_weights()

        input_ids = torch.randint(0, 1000, (2, 10), device=torch_device)

        with torch.no_grad():
            moe_output = moe_model(input_ids)
            dense_output = dense_model(input_ids)

        torch.testing.assert_close(
            dense_output.logits,
            moe_output.logits,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_equivalence_across_dtypes(self):
        """Test equivalence holds across different dtypes."""
        for dtype in [torch.float32, torch.bfloat16]:
            if dtype == torch.float16 and torch_device == "cpu":
                continue

            with self.subTest(dtype=dtype):
                moe_model, dense_model = self._create_models_with_shared_weights(dtype)

                input_ids = torch.randint(0, 1000, (1, 5), device=torch_device)

                with torch.no_grad():
                    moe_output = moe_model(input_ids)
                    dense_output = dense_model(input_ids)

                rtol = 1e-4 if dtype == torch.float32 else 1e-2
                atol = 1e-4 if dtype == torch.float32 else 1e-2

                torch.testing.assert_close(
                    dense_output.logits,
                    moe_output.logits,
                    rtol=rtol,
                    atol=atol,
                )
