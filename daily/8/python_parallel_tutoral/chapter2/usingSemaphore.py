from threading import Thread,Semaphore,currentThread
import random
import time

semaphore=Semaphore(0)

class Producer(Thread):
    def __init__(self):
        super().__init__(name='producer')
    def run(self):
        while(True):
            items=random.randint(1,5)
            print('produce',items)
            for i in range(items):
                semaphore.release()
            time.sleep(5)
class Customer(Thread):
    def __init__(self,name):
        super().__init__(name=name)
    def run(self):
        while(True):
            semaphore.acquire()
            print(currentThread().getName(),'achive items')
if __name__=='__main__':
    p=Producer()
    p.start()
    
    cs=[]
    for k in range(3):
        c=Customer('customer%d'%(k+1))
        c.start()
        cs.append(c)
    for c in cs:
        print('join')
        c.join()
    p.join()
