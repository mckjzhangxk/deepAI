from threading import Thread
import threading
from time import sleep

def sayHello(*args,**kwargs):
    print(args)
    sleep(2)
    #print(kwargs)
    T=threading.currentThread()
    print(T.getName(),' end')
threads=[]

for i in range(10):
    t=Thread(target=sayHello,args=(i,i*2,i**2),kwargs={'thread:':i})
    t.start()
##    t.join()  同步等待线程执行完成才会往下走
    threads.append(t)
for t in threads:
    t.join()
print('End processing')
