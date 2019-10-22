import numpy as np
from collections import defaultdict

def run_trails(items,prob,trails=10000,quitProb=.3):
    record=[]
    assert len(items)==len(prob)
    for _ in range(trails):
        chioceItems=set()
        while True:
            it=np.random.choice(items,p=prob)
            if it not in chioceItems:          
                chioceItems.add(it)
            if np.random.rand()<quitProb:
                record.append(tuple(chioceItems))
                break
    return record
def estimate(record,items):
    ret=defaultdict(int)
    for transaction in record:
        for item in transaction:
            ret[item]+=1
    sm=sum(ret.values())
    ret={key:value/sm for key,value in ret.items()}
    ret=[ret[e] for e in items]
    return ret
if __name__ == "__main__":
    items=['a','b','c','d']
    prob=[.7,.1,.1,.1]
    trails=10000

    for _ in range(30):
        record=run_trails(items,prob,trails)
        prob=estimate(record,items)
        print(prob)