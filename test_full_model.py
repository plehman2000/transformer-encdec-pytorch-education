import logging
from model import Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
import torch
import tiktoken
import settings

device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else "cpu")
encoder = tiktoken.get_encoding("o200k_base")


def encode(sequences_to_encode):
    batch = [
        torch.tensor(encoder.encode(x)) for x in sequences_to_encode
    ]  # ,# need to add padding
    batch = [
        seq % vocab_size for seq in batch
    ]  # ! need this because of tiny vocab for training purposes

    target_sequence = torch.full(
        size=(len(batch), max_seq_length), fill_value=pad_token_id, dtype=torch.long
    )
    for i in range(len(target_sequence)):
        target_sequence[i, : len(batch[i])] = batch[i]
    return target_sequence


max_seq_length = 100
vocab_size = 512
input_sequences_to_encode = [
    "My friend",
    "Bro this shit is crazy!",
    "What's up?",
]

output_sequences_to_encode = [
    "My wigger",
    "wigger this shi crazy",
    "Wus poppin?",
]


pad_token_id = (
    encoder.eot_token % vocab_size
)  #! add modulo for research purposes like above


input_ids = encode(input_sequences_to_encode)
input_ids = input_ids.to(device)

output_ids = encode(output_sequences_to_encode)
output_ids = output_ids.to(device)




model = Transformer(
    n_enc_blocks=1,
    n_dec_blocks=1,
    d_model=100,
    n_heads=2,
    d_head=10,
    vocab_size=vocab_size,
    max_seq_length=max_seq_length,
    pad_token_id=pad_token_id,
)



model.to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

iterations = 100
for _ in range(iterations):
    optim.zero_grad()
    output = model.forward(input_ids=input_ids, target_ids=output_ids
    )
    loss = criterion(output.view(-1, vocab_size), output_ids.view(-1))
    loss.backward()
    optim.step()

    logging.info(f"Loss: {loss.item():.4f}")
    # break







# # # Would be fruitful to build a vanilla transformer of a small scale, then run it on a small dataset
# # # then see how each archtectural change affects it
