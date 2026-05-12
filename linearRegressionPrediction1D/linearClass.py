import torch
from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features=1, out_features=1)

print(list(model.parameters()))

x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = model(x)
print(yhat)