import torch
import torch.nn as nn
import torch.nn.functional as F


a=torch.rand(10,9)
b=a>0.5
a.data.masked_fill_(b,2)
a.data[b]=3
print(a)
