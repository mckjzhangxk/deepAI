from threading import Lock,RLock

if __name__=='__main__':
##    lock=Lock()
    lock=RLock()
    
    lock.acquire()
    lock.acquire()

    print('Come Here')
    lock.release()
    lock.release()
