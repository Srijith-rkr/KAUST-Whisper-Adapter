from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .transcribe import transcribe as transcribe_function
from .decoding import detect_language as detect_language_function, decode as decode_function
from .tokenizer import Tokenizer, get_tokenizer


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def get_label_mapping_matrix():
    enclosed_tokenizer = get_tokenizer(multilingual =True)
    language_tokens_id = list(enclosed_tokenizer.all_language_tokens)
    label_mapping_matrix = torch.zeros(size=[1,51865,17],requires_grad=False) # might have to change required grad if you want to go back to non learnable matrix
    labels_per_dialect = len(language_tokens_id)/17 # happy mistake over here!! ; if it was // as planned before everything would have 5 and last would have 19
    
    for n,r in enumerate(language_tokens_id):
        c = int( (n-n%labels_per_dialect )/labels_per_dialect   )
        
        if c > label_mapping_matrix.shape[-1] - 1:
            c = label_mapping_matrix.shape[-1] - 1

        label_mapping_matrix[:,r,c] = 1 # dtype is float32 by defaule . dw :)
    return label_mapping_matrix



class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state) # MultiHeadAttention is attention with multiple heads. For the heads to be differnnet the Q K V matrices also have to be different. Hence we apply a liner trasnform on them
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        q_prev:  Optional[Tensor] = None,
    ):
        if q_prev:
            q = q_prev
        else:
            q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk, q = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk, q

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach(), q


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, add_adapter: bool = False, adapter_dim: int = 256, add_transformer_adapter: bool = False, layer_num: int = 0):
        super().__init__()
        
        self.add_adapter = add_adapter
        self.add_transformer_adapter = add_transformer_adapter
        self.adadpter_dim = adapter_dim
        self.n_state = n_state
        
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        self.mlp = nn.Sequential(Linear(n_state, n_state * 4), nn.GELU(), Linear(n_state * 4, n_state))
        self.mlp_ln = LayerNorm(n_state)
        
        if self.add_transformer_adapter:
            self.adapter_wte = nn.Embedding(adapter_dim, n_state)
            
        if self.add_adapter:
            self.adapter_ln_attn = LayerNorm(adapter_dim)
            self.adapter_ds = Linear(n_state, adapter_dim)
            self.adapter_gelu = nn.GELU()
            self.adapter_us = Linear(adapter_dim, n_state)
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        op, _, q = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        x = x + op
        if self.cross_attn:
            op, _, _ = self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
            x = x + op
        x = x + self.mlp(self.mlp_ln(x))
            
        if self.add_adapter: 
            x_down =  self.adapter_ds((x))
            x_gelu = self.adapter_gelu(x_ds)     
            x_up = self.adapter_us(x_gelu)
            x = x + x_up
            
        if self.add_transformer_adapter and not self.cross_attn : # Not activated for the decoder
            prefix = self.adapter_wte.weight.reshape(1, self.adadpter_dim, self.n_state).repeat(x.shape[0], 1, 1, 1)
            x_adapter_attn  =self.self.attn(prefix, q_prev = q)
            x = x + x_adapter_attn
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,add_adapter: bool = False, adapter_dim: int = 256, add_bridge: bool = False, add_transformer_adapter: bool = False):
        super().__init__()
        
        self.add_adapter = add_adapter
        self.adapter_dim = adapter_dim
        self.add_bridge = add_bridge
        self.add_transformer_adapter = add_transformer_adapter
        
        self.noise_matrix = nn.Parameter(  torch.normal(size = [80, 3000], mean=0.35, std=0.35),   requires_grad=True) # you handel the dtype insided the forward pass
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1) # torch.nn.Conv1d(in_channels, out_channels) convers from 80 channesl (from mel spectogram) to 512 channelsl (given by n_audio_state)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1) 
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        
        

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, add_adapter=self.add_adapter, adapter_dim=self.adapter_dim, layer_num= i, add_transformer_adapter = self.add_transformer_adapter) for i in range(n_layer)] # notice that cross attention in RAB is false in encoder but true in decoder 
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """ # x shape : 1 80 3000 is before the conovlution layers
        x = (x + self.noise_matrix).to(x.dtype)
        x = F.gelu(self.conv1(x)) # x shape : 1 512 3000
        x = F.gelu(self.conv2(x)) # # x shape : 1 512 1500 batch state/channels timestep
        x = x.permute(0, 2, 1) # batch timestep state/channels 1 1500 512

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        
        #tensor_list = []
        for block in self.blocks:
            # if self.add_bridge:
            #     x, tensor_list = block(x, tensor_List =tensor_list) # notice prev_rep_[X|x] # X is parameter name; x is to pass the arguments
            # else:
            x = block(x)
            

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state) #torch.nn.Embedding(num_embeddings, embedding_dim) SHOW
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)] # notice that cross attention in RAB is false in encoder but true in decoder 
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1) # looks like n_ctx is the number of words the modle can predict.  The mask is built for causual attention by having the upper triangle full of -inf values
        self.register_buffer("mask", mask, persistent=False)
        #self.adapter_label_mapping_matrix = nn.Parameter(get_label_mapping_matrix()  ,   requires_grad=True) ## named it adapter so that unfreezing would all ready be included with  current training code
        self.register_buffer("label_mapping_matrix", get_label_mapping_matrix(), persistent=True)
        
        

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, map_labels = False):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)    ISN'T IT BATCH, TIMESTEP, AUDIO STATE 
            the encoded audio features to be attended on
        """
        
        
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]] # adding token and positional embedding to the sot token
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        
        label_mapped_logits = logits @ self.label_mapping_matrix.to(x.dtype).to(x.device)
        
        if map_labels:
            return label_mapped_logits
        
        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, add_adapter: bool = False,adapter_dim: int = 256, add_bridge: bool = False, add_transformer_adapter: bool = False): #ModelDimensions is a dataclass that contains all the needed parameters as objects
        super().__init__()
        self.dims = dims
        self.add_adapter = add_adapter
        self.adapter_dim = adapter_dim
        self.add_bridge = add_bridge
        self.add_transformer_adapter = add_transformer_adapter
        
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            add_adapter=self.add_adapter,
            adapter_dim=self.adapter_dim,
            add_bridge=self.add_bridge,
            add_transformer_adapter = self.add_transformer_adapter
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
