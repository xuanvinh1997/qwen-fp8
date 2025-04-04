from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2MoeForCausalLM
import torch
import torch.nn.functional as F
from transformers import Qwen2MoeForCausalLM, Qwen2MoeConfig, Qwen2Model, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    Qwen2MLP,
)
from transformers import Qwen2MoeConfig
import torch.nn as nn


class Qwen2_5MoEConfig(Qwen2MoeConfig):
    model_type = "qwen2_5moe"

    def __init__(
        self,
        num_experts=3,
        top_k=1,
        capacity_factor=1.5,
        aux_loss_weight=0.01,
        router_jitter=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.router_jitter = router_jitter

        self.dtype = kwargs.get("dtype", torch.float32)
        self.dropout_rate = kwargs.get("dropout_rate", 0.01)


class Qwen2_5MoEExpertRouter(torch.nn.Module):
    """
    Mixture of Experts Router Layer

    Takes attention outputs as input and routes to top-k MLPs (experts)
    using a learned routing mechanism.
    """

    def __init__(
        self,
        input_dim,  # Dimension of input (from attention layer)
        num_experts,  # Total number of experts available
        top_k=1,  # Number of experts to route each token to
        capacity_factor=1.5,  # Scaling factor for expert capacity
        aux_loss_weight=0.01,  # Weight for auxiliary load balancing loss
        router_jitter=0.01,  # Optional noise to add during training
        dtype=torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.router_jitter = router_jitter
        self.dtype = dtype

        # Router projection layer: maps input to expert selection logits
        self.router = nn.Linear(input_dim, num_experts, bias=False, dtype=dtype)

        # Initialize with small weights to encourage equal expert usage early in training
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)

    def forward(self, inp, training=True):
        """
        Forward pass for the router

        Args:
            inp: Tensor of shape [batch_size, seq_len, input_dim] from attention layer
            training: Whether the model is in training mode

        Returns:
            dispatch_tensor: Sparse tensor for dispatching inputs to experts
            combine_tensor: Sparse tensor for combining expert outputs
            router_logits: Raw router logits
            aux_loss: Load balancing auxiliary loss
        """
        # Get shape information
        batch_size, seq_len, _ = inp.shape
        num_tokens = batch_size * seq_len

        # Reshape for routing
        inp_reshaped = inp.reshape(
            -1, self.input_dim
        )  # [batch_size * seq_len, input_dim]

        # Get router logits
        router_logits = self.router(inp_reshaped)  # [batch_size * seq_len, num_experts]
        # print("router_logits", router_logits)
        # Add jitter noise during training for stability
        if training and self.router_jitter > 0:
            router_logits += torch.randn_like(router_logits) * self.router_jitter

        # Calculate expert capacity: how many tokens can be routed to each expert
        # We scale by capacity_factor to allow for some experts to receive more tokens
        capacity = int(
            self.capacity_factor * num_tokens * self.top_k / self.num_experts
        )

        # Convert router logits to probabilities using softmax
        router_probs = F.softmax(
            router_logits, dim=-1
        )  # [batch_size * seq_len, num_experts]

        # Get top-k experts and their routing probabilities
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        print("top_k_probs", top_k_probs)
        print("top_k_indices", top_k_indices)
        # Normalize the top-k probabilities
        top_k_probs_sum = top_k_probs.sum(dim=-1, keepdim=True)
        top_k_probs_normalized = top_k_probs / top_k_probs_sum

        # Create mask for valid routing
        # Each token routes to top_k experts
        expert_mask = torch.zeros(
            num_tokens, self.num_experts, device=router_logits.device, dtype=torch.bool
        )

        # Create indices for scatter operation
        token_indices = (
            torch.arange(num_tokens, device=router_logits.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
        )

        # Populate the expert mask
        expert_mask.scatter_(1, top_k_indices, True)

        # Calculate auxiliary load balancing loss
        # We want to encourage all experts to be used equally
        # 1. Compute the fraction of router probability assigned to each expert
        router_prob_per_expert = router_probs.mean(0)

        # 2. Compute auxiliary loss: minimize the variance in expert utilization
        aux_loss = torch.mean(
            self.num_experts * router_prob_per_expert * router_prob_per_expert
        )

        # Create dispatch and combine tensors
        # These will be used to route inputs to experts and combine expert outputs

        # Create dispatch mask tracking which tokens go to which experts with what weights
        dispatch_tensor = torch.zeros(
            num_tokens, self.num_experts, device=router_logits.device, dtype=self.dtype
        )

        # For each token and its top-k experts, set the corresponding weight
        for token_idx in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k].item()
                prob = top_k_probs_normalized[token_idx, k].item()
                dispatch_tensor[token_idx, expert_idx] = prob

        # The combine tensor is the same as the dispatch tensor in this implementation
        # Some implementations might use different weights for combining
        combine_tensor = dispatch_tensor.clone()

        return {
            "dispatch_tensor": dispatch_tensor,
            "combine_tensor": combine_tensor,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "aux_loss": aux_loss,
            "top_k_indices": top_k_indices,
            "top_k_probs": top_k_probs_normalized,
        }

class Qwen2_5MoEDecoderLayer(torch.nn.Module):
    def __init__(self, config: Qwen2_5MoEConfig, layer_idx: int):
        super().__init__()
        self.input_dim = config.hidden_size
        self.output_dim = config.hidden_size
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        
        # Router and experts for MoE
        self.router = Qwen2_5MoEExpertRouter(
            input_dim=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            capacity_factor=config.capacity_factor,
            aux_loss_weight=config.aux_loss_weight,
            router_jitter=config.router_jitter,
        )

        self.experts = nn.ModuleList(
            [Qwen2MLP(config) for _ in range(config.num_experts)]
        )

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

   

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cos=None,
        sin=None,
        training=True
    ):
        """
        Forward pass for the MoE layer with proper handling of attention and cache
        """
        residual = hidden_states
        
        # First normalization before attention
        hidden_states = self.input_layernorm(hidden_states)
        # print("attention_mask", attention_mask)
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=(cos, sin),
        )
        # print("attn_outputs", attn_outputs)
        if use_cache:
            attn_output, past_key_value = attn_outputs[:2]
        else:
            attn_output = attn_outputs[0]
            past_key_value = None
            
        # Residual connection after attention
        hidden_states = attn_output + residual
        residual = hidden_states
        
        # Second normalization before MoE
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MoE Layer processing
        batch_size, seq_len, _ = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        # Get routing information
        router_outputs = self.router(hidden_states, training=training)
        dispatch_tensor = router_outputs["dispatch_tensor"]
        combine_tensor = router_outputs["combine_tensor"]
        aux_loss = router_outputs["aux_loss"]
        
        # Reshape input for expert processing
        inp_reshaped = hidden_states.reshape(-1, self.input_dim)
        
        # Initialize expert outputs
        expert_outputs = torch.zeros(
            num_tokens, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            expert_mask = dispatch_tensor[:, expert_idx] > 0
            if not expert_mask.any():
                continue
                
            # Select tokens for this expert
            expert_inputs = inp_reshaped[expert_mask]
            
            # Get expert weights for these tokens
            expert_weights = dispatch_tensor[expert_mask, expert_idx].unsqueeze(1)
            
            # Process inputs with the expert
            processed = expert(expert_inputs)
            
            # Weight the outputs by router probabilities
            weighted_outputs = processed * expert_weights
            
            # Add to the total outputs
            expert_outputs[expert_mask] += weighted_outputs
            
        # Reshape back to original dimensions
        moe_output = expert_outputs.reshape(batch_size, seq_len, self.output_dim)
        
        # Final residual connection
        hidden_states = moe_output + residual
        
        outputs = (hidden_states,)
        
        if output_attentions:
            # Add attention outputs if requested
            attentions = attn_outputs[-1]
            outputs = outputs + (attentions,)
            
        if use_cache:
            # Add past_key_value if caching is used
            outputs = outputs + (past_key_value,)
            
        # Add aux_loss as the last element
        outputs = outputs + (aux_loss,)
        
        return outputs

from dataclasses import dataclass
from typing import List, Optional, Tuple

from transformers.modeling_outputs import ModelOutput


@dataclass
class MoEModelOutputWithPast(ModelOutput):
    """
    Output class for MoE models.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        aux_loss (`torch.FloatTensor`, *optional*):
            Auxiliary load balancing loss for the MoE layers.
    """

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    aux_loss: Optional[torch.FloatTensor] = None

class Qwen2_5MoEModel(PreTrainedModel):
    def __init__(self, config: Qwen2_5MoEConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2_5MoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)


    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length=0):
        # Create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        
        if input_shape[-1] > 1:
            seq_length = input_shape[-1]
            
            # Create causal mask for auto-regressive training/generation
            causal_mask = torch.tril(
                torch.ones(
                    (seq_length, seq_length + past_key_values_length),
                    device=attention_mask.device,
                    dtype=dtype,
                )
            )
            # Convert to 4D: [1, 1, seq_length, seq_length + past_length]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Start with causal mask
            combined_attention_mask = causal_mask
            
            # Now handle padding mask if provided
            if attention_mask is not None:
                # Convert from [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
                # This will be broadcasted to the right shape when combined
                expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                
                # Convert from binary mask (1=attention, 0=no attention) to additive mask (0=attention, -inf=no attention)
                expanded_attn_mask = (1.0 - expanded_attn_mask.to(dtype)) * torch.finfo(dtype).min
                
                # Broadcast the padding mask to the same shape as causal mask
                # Because of broadcasting, we don't need to explicitly expand all dimensions
                combined_attention_mask = causal_mask + expanded_attn_mask
            else:
                # If no attention_mask provided, just use causal mask
                seq_length = input_shape[-1]
                combined_attention_mask = torch.ones(
                    (input_shape[0], seq_length, seq_length + past_key_values_length),
                    device=attention_mask.device,
                    dtype=dtype,
                )
                combined_attention_mask = torch.tril(combined_attention_mask)
                combined_attention_mask = combined_attention_mask.unsqueeze(1)
        
        return combined_attention_mask
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create embedding for input tokens if not provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        input_shape = (batch_size, seq_length)
        
        # Process attention_mask
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        # Compute rotary embeddings
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        # Process through each decoder layer
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        total_aux_loss = 0.0
        # Convert attention_mask to 4D format for attention mechanism
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds.dtype, past_key_values_length
        )
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Process through decoder layer with proper arguments
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cos=cos,
                sin=sin,
                training=self.training,
            )

            # First output is always hidden states
            hidden_states = layer_outputs[0]
            
            # Handle attention outputs if requested
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Handle cache if used
            if use_cache:
                # Cache is either at index 1 or 2 depending on whether attentions were output
                cache_index = 2 if output_attentions else 1
                next_decoder_cache += (layer_outputs[cache_index],)

            # Last output is always aux_loss
            total_aux_loss += layer_outputs[-1]

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attns,
                    total_aux_loss,
                ]
                if v is not None
            )

        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            aux_loss=total_aux_loss,
        )

class Qwen2_5MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen2_5MoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5MoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


from transformers.generation import GenerationMixin


class Qwen2_5MoEForCausalLM(Qwen2_5MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen2_5MoEConfig):
        super().__init__(config)
        self.model = Qwen2_5MoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids, attention_mask=None, position_ids=None
    ):
        
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def generate(
        self, input_ids, attention_mask=None, position_ids=None
    ):
        lm_logits = self(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # head_mask=head_mask,
        )
        # Implement your generation logic here
        # For example, you can use beam search or sampling to generate text
        # This is a placeholder for the actual generation logic
        generated_text = torch.argmax(lm_logits, dim=-1)
        return generated_text
    
    @staticmethod
    def merge_models(model_name_or_path: List[str]):
        """
        Merge the model with the specified model name or path.
        This is a placeholder for the actual merging logic.
        """
        config = Qwen2Config.from_pretrained(model_name_or_path[0])
        config = Qwen2_5MoEConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_experts=3,
            top_k=1,
            num_key_value_heads = 2

            # capacity_factor=1.5,
            # aux_loss_weight=0.01,
        )
        # Implement your merging logic here
        assert len(model_name_or_path) > 0, "At least one model name or path is required."
        if len(model_name_or_path) != config.num_experts:
            raise ValueError(
                f"Number of model names or paths ({len(model_name_or_path)}) must match the number of experts ({config.num_experts})."
            )


       
        # Initialize the main model with the MoE configuration
        moe_model = Qwen2_5MoEForCausalLM(config)
        experts = []
        for model_name in model_name_or_path:
            expert_model = AutoModelForCausalLM.from_pretrained(model_name)
            experts.append(expert_model)
        # Merge the experts into the main model
        moe_model.model.embed_tokens = experts[0].model.embed_tokens
        print("len(experts)", len(experts))
        print("len(moe_model.model.layers)", len(moe_model.model.layers))
        for i,layer in enumerate(moe_model.model.layers):
            if isinstance(layer, Qwen2_5MoEDecoderLayer):
                layer.self_attn = experts[0].model.layers[i].self_attn
                # layer.router # auto initialized
                layer.experts[i] = experts[i].model.layers[i].mlp
                layer.input_layernorm = experts[0].model.layers[i].input_layernorm
                layer.post_attention_layernorm = experts[0].model.layers[i].post_attention_layernorm
                pass
        moe_model.model.norm = experts[0].model.norm
        moe_model.model.rotary_emb = experts[0].model.rotary_emb
        moe_model.lm_head = experts[0].lm_head
        return moe_model