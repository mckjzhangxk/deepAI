from threading import Thread,Lock
N=800000
cnt_lock=0
cnk_unlock=0

res_lock=Lock()


def plus():
    global cnt_lock
    global N
    global res_lock
    for i in range(N):
        res_lock.acquire()
        cnt_lock+=1
        res_lock.release()

def minus():
    global cnt_lock
    global N
    global res_lock
    for i in range(N):
        res_lock.acquire()
        cnt_lock-=1
        res_lock.release()

def Uplus():
    global cnk_unlock
    global N
    for i in range(N):
        cnk_unlock+=1
def Uminus():
    global cnk_unlock
    global N
    for i in range(N):
        cnk_unlock-=1
      
t1=Thread(target=plus)
t2=Thread(target=minus)
t1.start()
t2.start()
t1.join()
t2.join()
print('With lock cnt={}'.format(cnt_lock))

t3=Thread(target=Uplus)
t4=Thread(target=Uminus)
t3.start()
t4.start()
t3.join()
t4.join()
print('Without lock cnt={}'.format(cnk_unlock))
