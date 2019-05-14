import torch
'''
scatter_ (dim,index,src)

if self have shape (d1,d2,...dm),dim=k

then index should have shape (d1,...dk-1,dU,dk+1,...,dm)
same shape with src


self[i1,i2,.index(i1,i2...im),im]=src[i1,i2,...im]

so you can see that this method copy all value from src
to self, 
'''
A=torch.rand(10,8,22,88)
dim=2
newAxis=4
index=torch.randint(0,22,(10,8,newAxis,88))
value=torch.ones((10,8,newAxis,88))
A.scatter_(dim,index,value)

