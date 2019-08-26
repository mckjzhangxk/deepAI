import numpy as np
def accuracy(yhat,y):
    p1=yhat.argmax().cpu().numpy()
    p2=y.cpu().numpy()
    acc=np.mean(p1==p2)
    return acc