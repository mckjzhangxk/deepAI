from threading import Thread
import threading
import time
from time import sleep

def printTime(count,delay):
    name=threading.currentThread().getName()
    for i in range(count):
        print('{}:{}'.format(name,time.ctime(time.time())))
        sleep(delay)

class MyThread(Thread):
    def __init__(self,name,delay):
        super().__init__(name=name)
        self.delay=delay
    def run(self):
        printTime(5,self.delay)


t1=MyThread('Thread 100',1)
t1.start()

t2=MyThread('Thread 200',5)
t2.start()


t1.join()
t2.join()
