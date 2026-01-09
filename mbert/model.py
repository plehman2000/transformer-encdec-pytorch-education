import torch
import torch.nn as nn
# >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
# >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
# >>> memory = torch.rand(10, 32, 512)
# >>> tgt = torch.rand(20, 32, 512)
# >>> out = transformer_decoder(tgt, memory)

shape = (10, 10)

x = torch.rand(shape)
tgt = torch.rand(shape)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(num_features=10))
        self.l2 = nn.Sequential(nn.ReLU(), nn.Linear(10, 10))

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        return out


model = SimpleModel()
optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

criterion = nn.HuberLoss()
its = 50

for _ in range(its):
    optim.zero_grad()
    out = model(x)
    loss = criterion(tgt, out)
    loss.backward()
    optim.step()
    print(f"Loss: {loss.item()}")

print(f"Final Loss after {its} iterations: {loss.item()}")
# No BN
# Final Loss after 50 iterations: 0.0828852429986

# With BN
# Final Loss after 50 iterations: 0.07897406071424484
