import os, signal
import argparse
def check_process(pstring):
    query = "ps aux | grep " + pstring + " | grep -v grep"
    pid=None
    for line in os.popen(query):
        fields = line.split()
        pid = fields[1]
        break
    return pid

def stopProcess(pstring):
    pid=check_process(pstring)
    if pid==None:
        print('服务没有开启')
    else:
        os.kill(int(pid), signal.SIGKILL)
        print('服务pid(%s)关闭成功' % pid)


def startProcess(pstring):
    pid=check_process(pstring)
    if pid==None:
        # os.system('nohup python3 -m cfai.cluster &')
        program = 'nohup'
        arguments = ('python3','-m','cfai.cluster')
        os.execvp(program, (program,) + arguments)
        print('服务开启成功')
    else:
        print('服务pid(%s)已经启动'%pid)
def showProcess(pstring):
    pid = check_process(pstring)
    if pid==None:

        print('服务没启动')
    else:
        print('服务pid(%s)已经启动'%pid)

def parse():
    parser=argparse.ArgumentParser()

    parser.add_argument('--start',action='store_true')
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--show', action='store_true')
    

    args=parser.parse_args()
    return args
if __name__ == '__main__':
    processname="'python3 -m cfai.cluster'"

    p=parse()
    if p.start:
        startProcess(processname)
    elif p.stop:
        stopProcess(processname)
    elif p.show:
        showProcess(processname)
    else:
        print('nothing to do')

