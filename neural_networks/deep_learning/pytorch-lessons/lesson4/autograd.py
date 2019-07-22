import torch 

x = torch.randn(2,2, requires_grad=True)

y =x**2

z = y.mean()
z.backward()
print(x.grad)
print(x/2)