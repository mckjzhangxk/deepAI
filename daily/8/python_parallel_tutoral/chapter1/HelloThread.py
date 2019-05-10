from threading import Thread
from time import sleep


class CookBook(Thread):
    def __init__(self,msg):
        super().__init__()
        self.message=msg
    def run(self):
        print('Thread start\n')
        for i in range(10):
            print(self.message)
            sleep(2)
        print('Thread end\n')

print('Processing start\n')
T1=CookBook('first thread')
T1.start()

T2=CookBook('second thread')
T2.start()

T1.join()
T2.join()
print('Processing end\n')
