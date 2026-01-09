import torch
import torch.nn as nn
import math
import logging
from einops import rearrange, repeat
import tiktoken

logging.basicConfig(level=logging.DEBUG)

### 1. **Vanilla scaled dot-product attention**

# Implement:
# [
# \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
# ]

# Focus on:


# * why scaling by √d matters
# * what softmax is actually doing to gradients
class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        # combines positional embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):
        # logging.debug(f"tokens.shape: {tokens.shape}")
        b, s = tokens.shape
        emb = self.emb(tokens)
        pos = torch.arange(s, device=tokens.device)
        pos = pos.unsqueeze(0).expand(b, s)  # probs a better way to do this
        pos = self.pos_emb(pos)
        return emb + pos


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        causal=True,
        temperature: float = 1.0,
        cross_attend=False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head
        self.causal = causal
        self.temperature = temperature
        self.cross_attend = cross_attend
        self.output_projection = nn.Linear(self.n_heads * self.d_head, self.dim)

    def forward(self, Q, K, V, causal_mask, padding_mask):
        logits = torch.matmul(Q, K.transpose(-2,-1)) / (
            math.sqrt(self.d_head) * self.temperature
        )  # scaling should be  depending on d head alone, not n heads

        # logging.info(f"logits.shape: {logits.shape}")
        if self.causal and causal_mask is not None:
            logits += causal_mask
        if padding_mask is not None:
            logits += padding_mask

        attn_probs = torch.softmax(logits, -1)

        attention_scores = torch.matmul(attn_probs, V)

        attention_scores = rearrange(attention_scores, "b n s d  -> b s (n d)")

        attn_output = self.output_projection(attention_scores)
        # logging(f"attention_scores: {attention_scores.size()}")
        return attn_output, attn_probs, logits


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model=10,
        n_heads=10,
        d_head=12,
        vocab_size=100,
        max_seq_length=100,
        pad_token=None,
        causal=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.causal = causal
        self.pad_token = pad_token

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ln3 = nn.LayerNorm(self.d_model)
        self.embedding = EmbeddingLayer(
            self.vocab_size, self.d_model, self.max_seq_length
        )

        self.Q_projection = nn.Linear(self.d_model, self.n_heads * self.d_head)
        self.K_projection = nn.Linear(self.d_model, self.n_heads * self.d_head)
        self.V_projection = nn.Linear(self.d_model, self.n_heads * self.d_head)

        self.multi_head_attn = MultiHeadAttention(
            dim=self.d_model, n_heads=self.n_heads, d_head=self.d_head
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )

    def forward(self, input: torch.Tensor):
        batch_size, seq_length = input.shape

        x = self.embedding(input)
        x = self.ln1(x)

        Q = rearrange(
            self.Q_projection(x).reshape(
                batch_size, seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )
        K = rearrange(
            self.K_projection(x).reshape(
                batch_size, seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )
        V = rearrange(
            self.V_projection(x).reshape(
                batch_size, seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )

        # Generate Causal Mask
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), fill_value=-1e9, device=x.device),
                diagonal=1,
            ).expand(batch_size, 1, seq_length, seq_length)
        # logging.info(f"causal_mask.shape: {causal_mask.shape}")
        # Generate Padding Mask

        padding_mask = get_padding_mask(input, self.pad_token)
        # padding mask = oadding mask with last dim duplicated
        # logging.info(f"padding_mask.shape: {padding_mask.shape}")

        attn_output, _, _ = self.multi_head_attn(Q, K, V, causal_mask, padding_mask)

        x = x + self.ln2(
            attn_output
        )  # attn being a group of weights, upweights the values of different inputs
        x = x + self.linear_layer(self.ln3(x))

        return x, input  # ! import


def get_padding_mask(input_ids, pad_token):
    batch_size, output_seq_length = input_ids.shape
    padding_locs = input_ids == pad_token  # True where padding
    padding_locs = padding_locs.unsqueeze(1).unsqueeze(2)  # get proper # of dimensions
    padding_locs = padding_locs.expand(
        batch_size, 1, output_seq_length, output_seq_length
    )

    padding_mask = torch.where(padding_locs, -1e9, 0.0)
    return padding_mask


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model=100,
        n_heads=2,
        d_head=10,
        vocab_size=100,
        max_seq_length=100,
        pad_token=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token

        self.embedding = EmbeddingLayer(
            self.vocab_size, d_model=self.d_model, max_seq_len=self.max_seq_length
        )

        self.LN1 = nn.LayerNorm(self.d_model)
        self.Q_projection_1 = nn.Linear(self.d_model, self.n_heads * self.d_head)
        self.K_projection_1 = nn.Linear(self.d_model, self.n_heads * self.d_head)
        self.V_projection_1 = nn.Linear(self.d_model, self.n_heads * self.d_head)

        self.masked_attn: MultiHeadAttention = MultiHeadAttention(
            dim=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            temperature=1.0,
            causal=True
        )

        self.Q_projection_2 = nn.Linear(self.d_model, self.n_heads * self.d_head)
        self.K_projection_2 = nn.Linear(self.d_model, self.n_heads * self.d_head)
        self.V_projection_2 = nn.Linear(self.d_model, self.n_heads * self.d_head)

        self.LN2 = nn.LayerNorm(self.d_model)

        self.cross_attention: MultiHeadAttention = MultiHeadAttention(
            dim=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            temperature=1.0,
            causal=False
        )

        self.LN3 = nn.LayerNorm(self.d_model)
        self.MLP = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

    def forward(self, input_embedding, input_ids, target_ids):
        batch_size, output_seq_length = target_ids.shape
        _, input_seq_length, _ = input_embedding.shape
        logging.info(f"input_embedding: {input_embedding.shape}")

        # * Embedding and Normalization
        output_embedding = self.embedding(target_ids)
        output_embedding = self.LN1(output_embedding)

        # * Q/K/V Projection of Target
        output_Q = rearrange(
            self.Q_projection_1(output_embedding).reshape(
                batch_size, output_seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )
        output_K = rearrange(
            self.K_projection_1(output_embedding).reshape(
                batch_size, output_seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )
        output_V = rearrange(
            self.V_projection_1(output_embedding).reshape(
                batch_size, output_seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )

        # * Generate Mask, Self Attention, apply mask to logits
        causal_mask = torch.triu(
            torch.full(
                (output_seq_length, output_seq_length),
                fill_value=-1e9,
                device=target_ids.device,
            ),
            diagonal=1,
        ).expand(batch_size, 1, output_seq_length, output_seq_length)

        padding_mask = get_padding_mask(target_ids, self.pad_token)

        output_embedding_attn, _, _ = self.masked_attn(
            output_Q, output_K, output_V, causal_mask, padding_mask
        )

        # * Residual Connection, pass output embedding forward unaltered
        decoder_hidden_state = output_embedding + output_embedding_attn

        # then
        decoder_hidden_state = self.LN2(decoder_hidden_state)

        decoder_hidden_state_Q = rearrange(
            self.Q_projection_2(decoder_hidden_state).reshape(
                batch_size, output_seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )

        input_hidden_state_K = rearrange(
            self.K_projection_2(input_embedding).reshape(
                batch_size, input_seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )
        input_hidden_state_V = rearrange(
            self.V_projection_2(input_embedding).reshape(
                batch_size, input_seq_length, self.n_heads, self.d_head
            ),
            "b s n d -> b n s d",
        )

        encoder_padding_mask = get_padding_mask(input_ids, self.pad_token)
        logging.info(f"encoder_padding_mask:  {encoder_padding_mask.shape}")

        cross_attn, _, _ = self.cross_attention(
            decoder_hidden_state_Q,
            input_hidden_state_K,
            input_hidden_state_V,
            None,
            encoder_padding_mask,
        )

        cross_attn_w_residual = cross_attn + decoder_hidden_state

        output = self.MLP(self.LN3(cross_attn_w_residual)) + cross_attn_w_residual
        return output


# ? Still need to make a wrapper for the layers, then a wrapper for enc/dec with the LM Head


# ? Learning notes:
"""
- have to pass input ids into the forward of the decoder so i can determine where to mask the inpout embeddings
- optionally crat e aomcmon projection so i dont adssume a commons eq length inpout to outptut

"""
