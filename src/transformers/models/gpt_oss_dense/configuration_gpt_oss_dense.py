# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""GptOssDense model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params


class GptOssDenseConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GptOssDenseModel`]. It is used to instantiate a
    GptOssDense model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer decoder.
        vocab_size (`int`, *optional*, defaults to 201088):
            Vocabulary size of the GptOssDense model.
        hidden_size (`int`, *optional*, defaults to 2880):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2880):
            Dimension of the MLP representations.
        head_dim (`int`, *optional*, defaults to 64):
            The attention head dimension.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for each attention layer. If None, defaults to num_attention_heads.
        sliding_window (`int`, *optional*, defaults to 128):
            Sliding window attention window size. If None, no sliding window is applied.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the MLP.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        rope_parameters (`dict`, *optional*):
            The RoPE parameters for the model. Defaults to yarn RoPE with specific parameters.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        layer_types (`list[str]`, *optional*):
            List of layer types (e.g., "sliding_attention", "full_attention"). If None, defaults to alternating pattern.
    """

    model_type = "gpt_oss_dense"
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "local_rowwise",
        "layers.*.mlp.gate_up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        num_hidden_layers: Optional[int] = 36,
        vocab_size: Optional[int] = 201088,
        hidden_size: Optional[int] = 2880,
        intermediate_size: Optional[int] = 2880,
        head_dim: Optional[int] = 64,
        num_attention_heads: Optional[int] = 64,
        num_key_value_heads: Optional[int] = 8,
        sliding_window: Optional[int] = 128,
        tie_word_embeddings: Optional[bool] = False,
        hidden_act: Optional[str] = "silu",
        initializer_range: Optional[float] = 0.02,
        max_position_embeddings: Optional[int] = 131072,
        rms_norm_eps: Optional[float] = 1e-5,
        rope_parameters: Optional[RopeParameters] = {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
        attention_dropout: Optional[float] = 0.0,
        use_cache: Optional[bool] = True,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.attention_bias = True
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 150000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["GptOssDenseConfig"]
