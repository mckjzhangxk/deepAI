import threading
import time


class Box:
    lock=threading.RLock()
    def __init__(self):
        self.total=0
    def execute(self,n):
        Box.lock.acquire()
        self.total+=n
        Box.lock.release()
    def add(self):
        self.execute(1)
    def remove(self):
        self.execute(-1)
def adder(box,items):
    for k in range(items):
        box.add()
        #time.sleep(5)
def remover(box,iems):
    for k in range(items):
        box.remove()
        #time.sleep(5)
if __name__=="__main__":
    items=5000000
    box=Box()

    t1=threading.Thread(target=adder,args=(box,items))
    t2=threading.Thread(target=remover,args=(box,items))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(box.total)
