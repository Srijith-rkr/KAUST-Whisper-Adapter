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
    ):
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

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

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
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, add_adapter: bool = False, adapter_dim: int = 256, add_bridge: bool = False, layer_num: int = 0 ):
        super().__init__()
        
        if add_bridge and (not add_adapter):
            print("cannot have briges without adapter")
            return
        
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))  #torch.nn.Linear(in_features, out_features, bias=True,
        self.mlp_ln = LayerNorm(n_state)
        
        self.add_adapter = add_adapter
        self.add_bridge = add_bridge
        self.layer_num = layer_num
#        self.temp = torch.zeros(size= [1, 1500, 512], requires_grad=False) # HARDCODED VALUES BE CAREFUL
                
        if self.add_adapter:
            # self.adapter_conv_on_cat = nn.Sequential( Conv1d(n_state*(self.layer_num+1), n_state, kernel_size=1, padding=0), nn.GELU())
            # self.adapter_conv_on_cat_ln =LayerNorm(n_state* (self.layer_num+1) )
            # self.adapter = nn.Sequential(Linear(n_state, adapter_dim), nn.GELU(), Linear(adapter_dim, n_state))
            # self.adapter_ln = LayerNorm(n_state)
            #self.prev_rep_X_ln = LayerNorm(n_state)

          #  self.adapter_ln_ds = LayerNorm(n_state)       
           # self.adapter_ln_us = LayerNorm(adapter_dim)    
            self.adapter_ln_attn = LayerNorm(adapter_dim)
            self.adapter_ds = Linear(n_state, adapter_dim)
            self.adapter_attn = MultiHeadAttention(adapter_dim, n_head) 
            self.adapter_gelu = nn.GELU()
            self.adapter_us = Linear(adapter_dim, n_state)

            

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
#        prev_rep_X: Optional[Tensor] = None,
       # tensor_List: list = [] ,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
            
            
        x = x + self.mlp(self.mlp_ln(x))
        
        # if self.add_bridge: 
        #     # temp = torch.zeros(size= x.shape, requires_grad=False).to(x.dtype).to(x.device)
        #     # for i in tensor_List:
        #     #     temp = temp + i
        #     if self.layer_num == 0 : x_ = x
        #     else :
        #         x_  = torch.cat(     (torch.cat(tensor_List,2),x),2 )
            
        #     x_ = self.adapter_conv_on_cat_ln(x_)
        #     x_ = x_.permute(0, 2, 1)
        #     x_ = self.adapter_conv_on_cat(x_)
        #     x_ = x_.permute(0, 2, 1)
            
        #     x = x + x_
            
        #     rep_X =  self.adapter( self.adapter_ln(x))       #was self.adapter( self.adapter_ln(x+ prev_rep_X)) + prev_rep_X  #was  self.adapter( self.adapter_ln(x)) + prev_rep_X   #was   self.adapter( self.adapter_ln(x + prev_rep_X)  )              #was self.adapter( self.adapter_ln(x) + self.prev_rep_X_ln(prev_rep_X ) ) where self.prev_rep_X_ln was non -trainable
        #     tensor_List.append(rep_X)
            
        #     x = x + rep_X
            
        #     return x , tensor_List
            
        if self.add_adapter: 
            x_ds =  self.adapter_ds((x))
            x_transform = self.adapter_attn(self.adapter_ln_attn(     x_ds          ))[0]
            x_ds = x_ds + x_transform
            x1 = self.adapter_gelu(x_ds)     
            x = x + self.adapter_us(x1)

           # x = x + self.adapter_us(    self.adapter_ln_us(self.adapter_attn(self.adapter_ln_attn(self.adapter_ds(self.adapter_ln_ds(x))))[0])     )
            
        return x  # return has been placed outside for the decoder

            
        


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,add_adapter: bool = False, adapter_dim: int = 256, add_bridge: bool = False):
        super().__init__()
        
        self.add_adapter = add_adapter
        self.adapter_dim = adapter_dim
        self.add_bridge = add_bridge
        
        self.noise_matrix = nn.Parameter(  torch.normal(size = [80, 3000], mean=0.35, std=0.35),   requires_grad=True) # you handel the dtype insided the forward pass
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1) # torch.nn.Conv1d(in_channels, out_channels) convers from 80 channesl (from mel spectogram) to 512 channelsl (given by n_audio_state)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1) 
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        
        

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, add_adapter=self.add_adapter, adapter_dim=self.adapter_dim, add_bridge=self.add_bridge, layer_num= i) for i in range(n_layer)] # notice that cross attention in RAB is false in encoder but true in decoder 
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
    def __init__(self, dims: ModelDimensions, add_adapter: bool = False,adapter_dim: int = 256, add_bridge: bool = False): #ModelDimensions is a dataclass that contains all the needed parameters as objects
        super().__init__()
        self.dims = dims
        self.add_adapter = add_adapter
        self.adapter_dim = adapter_dim
        self.add_bridge = add_bridge
        
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            add_adapter=self.add_adapter,
            adapter_dim=self.adapter_dim,
            add_bridge=self.add_bridge
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
