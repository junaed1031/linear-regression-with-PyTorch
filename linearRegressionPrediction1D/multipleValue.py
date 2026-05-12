import torch

w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([-1.0], requires_grad=True)

def forward(x):
    y = w*x+b
    return y


#multi value x = torch.tensor([1.0, 2.0, 3.0])

x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = forward(x)
print(yhat)