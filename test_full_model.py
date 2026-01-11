import logging

import tiktoken
import torch

from model import Transformer

device = torch.device("cuda")  # "cuda" if torch.cuda.is_available() else "cpu")
encoder = tiktoken.get_encoding("o200k_base")


def encode(sequences_to_encode):
    batch = [
        torch.tensor(encoder.encode(x)) for x in sequences_to_encode
    ]  # ,# need to add padding
    # batch = [
    #     seq % vocab_size for seq in batch
    # ]  # ! need this because of tiny vocab for training purposes

    target_sequence = torch.full(
        size=(len(batch), max_seq_length), fill_value=PAD_ID, dtype=torch.long
    )
    for i in range(len(target_sequence)):
        target_sequence[i, : len(batch[i])] = batch[i]
    return target_sequence

def make_decoder_inputs_and_targets(output_ids, bos_id, eos_id, pad_id):
    batch_size, seq_len = output_ids.shape
    decoder_inputs = torch.full_like(output_ids, fill_value=bos_id)
    decoder_inputs[:, 1:] = output_ids[:, :-1]  # shift right

    # Make decoder_targets same as output_ids, but ensure EOS at the last real token
    decoder_targets = output_ids.clone()
    for i in range(batch_size):
        # find the length of actual sequence (non-pad)
        seq_length = (output_ids[i] != pad_id).sum().item()
        if seq_length < seq_len:
            decoder_targets[i, seq_length] = eos_id  # put EOS at the first PAD
        else:
            decoder_targets[i, -1] = eos_id  # or last position if fully filled
    return decoder_inputs, decoder_targets


max_seq_length = 100
vocab_size = encoder.n_vocab

input_sequences_to_encode = [
    "Hello",
]

output_sequences_to_encode = [
    "Hola"
]


PAD_ID = vocab_size
BOS_ID = vocab_size + 1
EOS_ID = encoder.eot_token      # keep end-of-text as EOS
vocab_size += 2                 # extend vocab for PAD and BOS


input_ids = encode(input_sequences_to_encode)
input_ids = input_ids.to(device)

output_ids = encode(output_sequences_to_encode)
output_ids = output_ids.to(device)


model = Transformer(
    n_enc_blocks=2,
    n_dec_blocks=2,
    d_model=100,
    n_heads=2,
    d_head=10,
    vocab_size=vocab_size,
    max_seq_length=max_seq_length,
    pad_token_id=PAD_ID,
)

PATH = "model_weights.pth"

# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint)

model.to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)

optim = torch.optim.AdamW(model.parameters(), lr=5e-4)

iterations = 50
for i in range(iterations):
    optim.zero_grad()

    decoder_inputs, decoder_targets = make_decoder_inputs_and_targets(
    output_ids, PAD_ID, EOS_ID, PAD_ID
    )

    output = model(input_ids, decoder_inputs)
    loss = criterion(
    output.view(-1, vocab_size),
    decoder_targets.view(-1),
    )


    # output = model.forward(input_ids=input_ids, target_ids=output_ids)
    # loss = criterion(output.view(-1, vocab_size), output_ids.view(-1))
    loss.backward()
    optim.step()

    logging.info(f"{i} Loss: {loss.item():.4f}")
    # if loss.item() < 0.01:
    #     break
torch.save(model.state_dict(), PATH)

# inference

# input_text = "My friend"
max_output_length = 100
model.eval()

input_text = "Hello"
while True:
    input_text = input("Chat: ")

    with torch.no_grad():
        input_ids = encode([input_text]).to(device)

        decoder_ids = torch.tensor([[BOS_ID]], device=device)

        for _ in range(max_output_length):
            model_out = model(input_ids, decoder_ids)
            logits = model_out[:, -1, :]
            next_id = torch.argmax(logits, dim=-1)

            decoder_ids = torch.cat([decoder_ids, next_id.unsqueeze(1)], dim=1)

            if next_id.item() == EOS_ID:  # stop at EOS
                break

        output_text = encoder.decode(decoder_ids[0, 1:].tolist())

        print(output_text)


