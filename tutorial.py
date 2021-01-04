import torch  #

a = torch.randn(16, 3, 32, 32)

print(a.size()) 
# torch.Size([16, 3, 32, 32])

b = a.view(-1) # view is faster than flatten

print(b.size()) 
# torch.Size([49152]) = 16*3*32*32

c = a.view(16, -1, 32)
print(c.size()) 
# torch.Size([16, 96, 32])

d = a.view(-1, 16, 32)
print(d.size())
# torch.Size([96, 16, 32])

print(a[0, 0, 0, 0])
# tensor(-0.3249)

print(b[0])
# tensor(-0.3249)

a[0, 0, 0, 0] = 3.0
print(a[0, 0, 0, 0])
# tensor(3.)

print(b[0])
# tensor(3.)

c = torch.randn(1, 10) # 1 row x 10 cols
print(c.size()) 
# torch.Size([1, 10])

d = c.squeeze(0) # 1, 10 -> 10,
print(d.size()) 
# torch.Size([10])

e = c.unsqueeze(-1)
print(e.size()) 
# torch.Size([1, 10, 1])

f = e.squeeze(2)
print(f.size()) 
# torch.Size([1, 10])

g = torch.randn(32, 1, 10, 1, 42, 1, 52)
print(g.size())
# torch.Size([32, 1, 10, 1, 42, 1, 52])

h = g.squeeze()
print(h.size())
# torch.Size([32, 10, 42, 52])

i = torch.arange(10, 11)
print(i) # tensor([10])

i.clamp(min=0) # relu
print(i)

j = torch.clamp(i, min=0, max=8, out=i**2)


