import torch
import torch.nn as nn
from transformers import ViTConfig
import math
from performer_pytorch import SelfAttention as PerformerAttention

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask = None, output_attentions: bool = False
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class MultiplicativeAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask = None, output_attentions: bool = False
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class AdditiveAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.score = nn.Linear(self.attention_head_size, 1, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask = None, output_attentions: bool = False
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        query_layer = query_layer.unsqueeze(2)
        key_layer = key_layer.unsqueeze(3)
        attention_scores = self.score(torch.tanh(query_layer + key_layer)).squeeze(-1)

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class CustomAttention(nn.Module):
    def __init__(self, config: ViTConfig, att) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.score = nn.Linear(self.attention_head_size, 1, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # combination
        input_size = 591  # Assuming two attention maps with num_channels channels each
        self.weighted_sum_network = WeightedSumNetwork(input_size)
        self.att = att

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask = None, output_attentions: bool = False
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # multiplicative attention
        attention_scores1 = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs1 = nn.functional.softmax(attention_scores1, dim=-1)

        # scaled attention
        attention_scores2 = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_probs2 = nn.functional.softmax(attention_scores2, dim=-1)

        # additive
        query_layer3 = query_layer.unsqueeze(2)
        key_layer3 = key_layer.unsqueeze(3)
        attention_scores3 = self.score(torch.tanh(query_layer3 + key_layer3)).squeeze(-1)
        attention_probs3 = nn.functional.softmax(attention_scores3, dim=-1)

        # fusion
        if self.att == 4:
            attention_probs = torch.mul(attention_probs1, attention_probs2)
        elif self.att == 5:
            attention_probs = torch.maximum(attention_probs1, attention_probs2)
        elif self.att == 6:
            attention_probs = torch.add(attention_probs1, attention_probs2)/2
        elif self.att == 7:
            attention_probs = torch.mul(attention_probs1, attention_probs3)
        elif self.att == 8:
            attention_probs = torch.maximum(attention_probs1, attention_probs3)
        elif self.att == 9:
            attention_probs = torch.add(attention_probs1, attention_probs3)/2
        elif self.att == 10:
            attention_probs = torch.mul(attention_probs2, attention_probs3)
        elif self.att == 11:
            attention_probs = torch.maximum(attention_probs2, attention_probs3)
        elif self.att == 12:
            attention_probs = torch.add(attention_probs2, attention_probs3)/2
        elif self.att == 13:
            attention_probs = torch.mul(torch.mul(attention_probs1, attention_probs2), attention_probs3)
        elif self.att == 14:
            attention_probs = torch.maximum(torch.maximum(attention_probs1, attention_probs2), attention_probs3)
        elif self.att == 15:
            attention_probs = torch.add(torch.add(attention_probs1, attention_probs2), attention_probs3)/3
        elif self.att == 16: # Combine the attention maps using the learned weighted sum
            attention_probs = self.weighted_sum_network(attention_probs1, attention_probs2, attention_probs3)
        elif self.att == 17:
            attention_probs = torch.mean(torch.stack([attention_probs1, attention_probs2, attention_probs3]), dim=0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, num_feats=256, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0., share_kv=False):
        super().__init__()
        assert (dim % num_heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.num_feats = num_feats

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, num_feats)))
        if share_kv:
            self.proj_v = self.proj_k
        else:
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, num_feats)))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, head_mask = None, output_attentions: bool = False):
        b, n, d = x.shape
        d_h, h, k = self.head_dim, self.num_heads, self.num_feats
        kv_len = n
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.scale * self.query(x).reshape(b, n, h, d_h).transpose(1, 2)
        kv = self.kv(x).reshape(b, n, 2, d).permute(2, 0, 1, 3)
        keys, values = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        # project keys and values along the sequence length dimension to k
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)
        kv_projs = (self.proj_k, self.proj_v)
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(
            1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        attn = torch.einsum('bhnd,bhkd->bhnk', queries, keys)
        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return (out,)

class PerformerWrapper(nn.Module):
    def __init__(self):
        super(PerformerWrapper, self).__init__()
 
    def forward(self, x, head_mask = None, output_attentions: bool = False):
        attn = PerformerAttention(dim = 768, heads = 12, causal = False).cuda()
        return (attn(x),)

class WeightedSumNetwork(nn.Module):
    def __init__(self, input_size):
        super(WeightedSumNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 3)  # A single output representing the weight
    
    def forward(self, attention_map1, attention_map2, attention_map3):
        # Concatenate the attention maps along the channel dimension
        combined_attention = torch.cat((attention_map1, attention_map2, attention_map3), dim=-1)
        
        # Pass the combined attention maps through the network
        weights = self.fc(combined_attention)
        # Apply a sigmoid activation to ensure weights are between 0 and 1
        weights = torch.softmax(weights, dim=-1)
        # Apply the learned weights to combine the attention maps
        combined_attention_map = weights[:,:,:,0].unsqueeze(-1) * attention_map1 + weights[:,:,:,1].unsqueeze(-1) * attention_map2 + weights[:,:,:,2].unsqueeze(-1) * attention_map3
        
        return combined_attention_map