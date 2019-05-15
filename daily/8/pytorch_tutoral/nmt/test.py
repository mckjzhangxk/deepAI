import torch


x=torch.rand(10,1)
y=torch.rand(10,1)
z=torch.cat((x,y),1)
print(z.size())
