import logging
import os
import random

import numpy as np
import tiktoken
import torch

from model import TransformerEncoderLayer

import settings

iterations = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = tiktoken.get_encoding("o200k_base")

max_seq_length = 100
# vocab_size = encoder.n_vocab # this is what we use IRL
vocab_size = 512


sequences_to_encode = [
    "Michael Lehman",
    "Bro this shit is crazy!",
    "Fortnite burger six seven?",
]

batch = [
    torch.tensor(encoder.encode(x)) for x in sequences_to_encode
]  # ,# need to add padding
batch = [
    seq % vocab_size for seq in batch
]  # ! need this because of tiny vocab for training purposes

pad_token_id = (
    encoder.eot_token % vocab_size
)  #! add modulo for research purposes like above
logging.debug(f"encoder.n_vocab: {encoder.n_vocab}")
print(f"The EOT(I will use as pad) token ID is: {pad_token_id}")


model = TransformerEncoderLayer(
    d_model=100,
    n_heads=10,
    d_head=10,
    vocab_size=vocab_size,
    max_seq_length=max_seq_length,
    pad_token=pad_token_id,
)
model.to(device)
criterion = torch.nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)


batch_padded = torch.full(
    size=(len(batch), max_seq_length), fill_value=pad_token_id, dtype=torch.long
)

for i in range(len(batch_padded)):
    batch_padded[i, : len(batch[i])] = batch[i]
batch_padded = batch_padded.to(device)

# batch_padded = torch.concat(batch_padded)
logging.info(f"batch_padded: {batch_padded.shape}")

target_seq = torch.randn(3, 100, 100, device=device)  # (batch, seq_len, dim)

for _ in range(iterations):
    optim.zero_grad()
    output, input = model(batch_padded)
    loss = criterion(output, target_seq)
    loss.backward()
    optim.step()

    logging.info(f"Loss: {loss.item():.4f}")
    break
